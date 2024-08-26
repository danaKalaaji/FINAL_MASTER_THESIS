from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM
import argparse
from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import base64
import torch
import librosa
import io
import pandas as pd

app = Flask(__name__)
CORS(app)
app.json.sort_keys = False


#### Functions to be used if you want to store the logits in a JSON file and load them back ###
# Convert tensors to lists for JSON serialization (because JSON cannot serialize tensors)
def convert_tensors_to_lists(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_tensors_to_lists(i) for i in obj]
    else:
        return obj
# Convert lists to tensors for JSON serialization (useful if you had them stored as lists)
def convert_lists_to_tensors(obj):
    if isinstance(obj, list):
        try:
            # Try to convert the list to a tensor
            return torch.tensor(obj)
        except (ValueError, TypeError):
            # If the list cannot be converted to a tensor, convert its elements instead
            return [convert_lists_to_tensors(i) for i in obj]
    else:
        return obj


def wav2vec2(audio, sr, processor, model, hotwords=None, output_word_offsets=True, 
             output_char_offsets=True, phoneme=False, start_offset=None, end_offset=None):
    """
    Transcribes the given audio using the wav2vec2 model.

    Args:
        audio (np.ndarray): The audio data as a numpy array.
        sr (int): The sample rate of the audio.
        processor (Wav2Vec2Processor): The processor used to preprocess the audio.
        model (Wav2Vec2ForCTC): The wav2vec2 model used for transcription.
        hotwords (List[str], optional): A list of hotwords used for decoding. Defaults to None.
        output_word_offsets (bool, optional): Whether to output word offsets during decoding. Defaults to True.
        output_char_offsets (bool, optional): Whether to output character offsets during decoding. Defaults to True.
        phoneme (bool, optional): Whether to perform phoneme decoding. Defaults to False.
        start_offset (float, optional): The start offset in seconds for audio processing. Defaults to None.
        end_offset (float, optional): The end offset in seconds for audio processing. Defaults to None.

    Returns:
        transcription_dict (dict): A dictionary containing the (transcription and) offsets.
        logits (torch.Tensor): The logits obtained from the model.
    """
    
    # If start_offset and end_offset are not provided, use the entire audio file 
    start_offset = start_offset if start_offset else 0
    end_offset = end_offset if end_offset else len(audio)/sr
    audio = audio[int(start_offset * sr):int(end_offset * sr)]
    
    # Process the audio file with the model and get the logits
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits.cpu()
    
    if phoneme:
        # Decode the logits to get the transcription and the character offsets (asr_phonemizer)
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription_dict = processor.decode(predicted_ids[0], output_char_offsets = output_char_offsets)
        transcription_dict = {#"text": transcription_dict['text'].lower(), #kept it in case we want to use it
                             "char_offsets": transcription_dict['char_offsets'] if output_char_offsets else None }

    else:
        # Decode the logits to get the transcription and the word offsets (asr_segmenter)
        transcription = processor.batch_decode(logits.numpy(), output_word_offsets=output_word_offsets, 
                                                 hotwords=hotwords) # note: can change hotword_weight here if needed
                                            
        # compute `time_offset` in seconds as product of downsampling ratio and sampling_rate
        time_offset = model.config.inputs_to_logits_ratio / processor.feature_extractor.sampling_rate
        for dict in transcription.word_offsets[0]:
            dict["start_offset"] = dict["start_offset"] * time_offset
            dict["end_offset"] = dict["end_offset"] * time_offset

        transcription_dict = {#"text": transcription.text[0], #kept it in case we want to use it
                            "word_offsets": transcription.word_offsets[0] 
                            }
    return transcription_dict, logits

def segment_speech(word_offsets, target_duration, min_duration, overlap=False):
    """
    Segments speech based on word offsets and target duration to have a certain amount of actual speech
    per segment.

    Args:
        word_offsets (list): A list of dictionaries representing the start and end offsets of words.
        target_duration (float): The desired duration of each segment.
        min_duration (float): The minimum duration allowed for a segment.
        overlap (bool, optional): Whether to allow overlapping segments. Defaults to False.

    Returns:
        list: A list of dictionaries representing the start and end offsets of each segment.
    """
    segments = []
    current_segment = []
    current_duration = 0.0
    
    # If the total duration of the words is less than the target duration, return the entire segment
    if word_offsets[-1]['end_offset'] - word_offsets[0]['start_offset'] < target_duration:
        return [{'start_offset': word_offsets[0]['start_offset'], 'end_offset': word_offsets[-1]['end_offset']}]
    
    # Split the words into segments based on the target duration
    for word in word_offsets:
        word_duration = word['end_offset'] - word['start_offset']
        if current_duration + word_duration > target_duration:
            current_segment.append(word)
            segments.append(current_segment)
            current_segment = [word] if overlap else []
            current_duration = word_duration if overlap else 0.0
        else:
            current_segment.append(word)
            current_duration += word_duration
    # Add the last segment if it is longer than the minimum duration
    if current_segment:
        if current_duration < min_duration and segments:
            last_segment = segments.pop()
            merged_segment = last_segment + current_segment
            split_index = len(merged_segment) // 2
            segments.append(merged_segment[:split_index])
            segments.append(merged_segment[split_index:])
        else:
            segments.append(current_segment)
    # Get the start and end offsets of each segment
    segment_intervals = []
    for segment in segments:
        start_offset = segment[0]['start_offset']
        end_offset = segment[-1]['end_offset']
        segment_intervals.append({'start_offset': start_offset, 'end_offset': end_offset})

    return segment_intervals


def run_models(audio, target_duration_segment, min_duration_segment, overlap, reference_text, processor_phoneme, 
               model_phoneme, processor_asr, model_asr, segmenter=True):
    """
    Run models to process audio and generate transcriptions, logits, and character offsets.

    Args:
        audio (numpy.ndarray): The audio data as a 1-D numpy array.
        target_duration_segment (float): The target duration of each segment in seconds.
        min_duration_segment (float): The minimum duration of each segment in seconds.
        overlap (bool): Whether to allow overlapping segments.
        reference_text (str): The reference text used for hotword.
        processor_phoneme (Processor): The processor for the phoneme model.
        model_phoneme (Wav2Vec2ForCTC): The phoneme model.
        processor_asr (Processor): The processor for the ASR model.
        model_asr (Wav2Vec2ForCTC): The ASR model.
        segmenter (bool, optional): Whether to use the segmenter. Defaults to True.

    Returns:
        tuple: A tuple containing the transcriptions, logits, and character offsets of the segments.
    """
    transcriptions_segments = []
    char_offsets_segments = []
    logits_segments = []

    audio_length = librosa.get_duration(y=audio, sr=16000)
    if audio_length < target_duration_segment + min_duration_segment:
        segmenter = False

    if segmenter:
        transcription_asr_segmenter, logits_asr = wav2vec2(audio, 16000, processor=processor_asr, model=model_asr, 
                                                hotwords=reference_text.split(), output_word_offsets=True, 
                                                alpha=0.0, beta=0.0)
        word_offsets = transcription_asr_segmenter['word_offsets']     
        for entry in segment_speech(word_offsets, target_duration_segment, min_duration_segment, overlap):
            onset = max(0, entry['start_offset'] - 1)
            offset = min(audio_length, entry['end_offset'] + 1)
            transcription_clip_all, logits_clip_all = wav2vec2(audio, 16000, 
                                                               processor=processor_phoneme, model=model_phoneme, 
                                                               start_offset=onset, end_offset=offset, 
                                                               phoneme=True, output_char_offsets=True)

            #transcriptions_segments.append(transcription_clip_all['text'])
            char_offsets_segments.append(transcription_clip_all['char_offsets'])
            logits_segments.append(logits_clip_all)
    else:
        transcription_clip_all, logits_clip_all = wav2vec2(audio, 16000, processor=processor_phoneme, model=model_phoneme, 
                                                            phoneme=True, output_char_offsets=True)
        #transcriptions_segments.append(transcription_clip_all['text'])
        char_offsets_segments.append(transcription_clip_all['char_offsets'])
        logits_segments.append(logits_clip_all)

    return transcriptions_segments, logits_segments, char_offsets_segments



def split_logit_matrix(logits_segment, char_offsets_segment):
    """
    Splits the logit matrix based on the character offsets.

    Args:
        logits_segment (numpy.ndarray): The logit matrix to be split.
        char_offsets_segment (list): A list of dictionaries containing character offsets.

    Returns:
        list: A list of sub-logit matrices, each corresponding to an item delimited by spaces.
    """
    # get offsets of all ' ' characters in the char_offsets_segments dict
    space_offsets = [(item['start_offset'], item['end_offset']) for item in char_offsets_segment if item['char'] == ' ']
    if not space_offsets:
        space_offsets = [(0, len(char_offsets_segment))]

    # cut the logits and the prediction at the first space
    sub_logits_segments = []
    prev_offset = 0
    for offset in space_offsets:
        # change size of logits_segment0 to the size of the first segment
        sub_logits_segment = logits_segment[:, prev_offset:offset[1]]
        prev_offset = offset[0]
        sub_logits_segments.append(sub_logits_segment)
            
    return sub_logits_segments


def multiple_predictions(sub_logits_segment, processor_phoneme, threshold_diff, 
                         max_pred_per_word, k, pad_token_id, space_token_id):
    """
    Generate multiple predictions for each timeframe based on alternative predictions.

    Args:
        sub_logits_segment (torch.Tensor): Sub-logits matrix for an item.
        processor_phoneme (Processor): Processor for decoding predictions.
        threshold_diff (float): Threshold difference between top-1 and alternative predictions.
        max_pred_per_word (int): Maximum number of predictions per item.
        k (int): Number of alternative phonemes to consider per timeframe.
        pad_token_id (int): ID of the pad token.
        space_token_id (int): ID of the space token.

    Returns:
        list: List of decoded predictions.
    """
    # Get top-k values and indices
    topk_values, topk_indices = torch.topk(sub_logits_segment, k=k, dim=-1)
    top1_indices = topk_indices[0, :, 0].tolist()

    # Initialize with the top-1 prediction
    predictions = [tuple(top1_indices)]

    # Iterate over each alternative prediction for each timeframe
    for j in range(1, k):
        for i in range(len(top1_indices)):  

            top1_idx = topk_indices[0][i][0].item()
            top1_value = topk_values[0][i][0].item()
            alt_idx = topk_indices[0][i][j].item()
            alt_value = topk_values[0][i][j].item()

            # If the top1 prediction is a pad token and the alternative is a space token or vice versa, skip (no need to swap)
            if not ((top1_idx == pad_token_id and alt_idx == space_token_id) 
                    or (top1_idx == space_token_id and alt_idx == pad_token_id)):
                
                diff = top1_value - alt_value
                if diff < threshold_diff:
                    new_predictions = []
                    for pred in predictions:
                        if len(predictions) >= max_pred_per_word + 1:
                            break
                        new_prediction = list(pred)
                        new_prediction[i] = topk_indices[0][i][j]
                        new_predictions.append(tuple(new_prediction))
                    # Add new predictions and remove duplicates
                    predictions.extend(new_predictions)
                    predictions = list(set(predictions))
            # Stop if we have reached the max prediction count
            if len(predictions) > max_pred_per_word:
                break

    predictions = [torch.tensor(prediction) for prediction in predictions]
    decoded_predictions = processor_phoneme.batch_decode(predictions)
    decoded_predictions = list(set(decoded_predictions))
    return decoded_predictions


def decode_phonetic_prediction_concat_error(all_decoded_predictions, target_phonemes, error_list, correct, results_decode):
    """
    Decodes the phonetic predictions by concatenating previous, current, and next predictions,
    and checks for errors in the concatenated predictions.

    Args:
        all_decoded_predictions (list): List of phonetic predictions for each segment.
        target_phonemes (list): List of target phonemes for each target item in the reference text.
        error_list (list): List of error phonemes for each target item in the reference text.
        correct (list): List to store the score of each target.
        results_decode (list): List to store the decoded predictions.

    Returns:
        correct (list): List of correctness for each target phoneme.
        all_decoded_predictions (list): List of phonetic predictions for each time step.
        results_decode (list): List of decoded predictions.
    """
    # Concatenate previous, current, and next predictions
    for j in range(1, len(all_decoded_predictions) -1):
        previous_pred_list = all_decoded_predictions[j-1]
        current_pred_list = all_decoded_predictions[j]
        next_pred_list = all_decoded_predictions[j+1]
        decoded_prediction_concat = []
        for previous_pred in previous_pred_list:
            for current_pred in current_pred_list:
                for next_pred in next_pred_list:
                    concat_pred = previous_pred + current_pred + next_pred
                    decoded_prediction_concat.append(concat_pred)

        # Binary classification of each target
        for i, targets_words in enumerate(target_phonemes):
            for concat_pred in decoded_prediction_concat:
                if any(error in concat_pred for error in error_list[i]):
                    correct[i] = 0
                    break
                for target_word in targets_words:
                    if not correct[i] and target_word in concat_pred:
                        correct[i] = 1
                        results_decode[i] = [concat_pred]
                        break

    return correct, all_decoded_predictions, results_decode            


def pipeline_asr_function(audio, reference_text, model_asr, processor_asr, model_phoneme, processor_phoneme, 
                 df_targets, target_duration_segment, min_duration_segment, threshold_diff, max_pred_per_word, 
                 k, pad_token_id, space_token_id, overlap, segmenter):
    """
    call the asr pipeline on the given audio using the provided models and processors.

    Args:
        audio (1D numpy array):Audio file.
        reference_text (str): Reference text.
        model_asr: ASR segmenter model .
        processor_asr: ASR segmenter processor.
        model_phoneme: ASR phonemizer model.
        processor_phoneme: ASR phonemizer processor.
        df_targets: DataFrame containing target words and their corresponding phonemes and error lists.
        target_duration_segment (float): Target duration for each speech segment.
        min_duration_segment (float): Minimum duration for each speech segment.
        threshold_diff (float): Threshold difference for phoneme predictions.
        max_pred_per_word (int): Maximum number of predictions per item.
        k (int): Maximum number of phonemes to consider for each timeframe.
        pad_token_id (int): ID of the padding token.
        space_token_id (int): ID of the space token.
        overlap (bool): Whether to allow overlapping segments.
        segmenter (bool): Whether to use the segmenter model.

    Returns:
        correct (list): List of binary values indicating whether each word in the reference text was correctly predicted.
        results_decode (list): List of decoded phonetic predictions for each word in the reference text.
    """
 
    reference_words = reference_text.split(" ")
    target_phonemes = []
    error_list = []
    correct = [0] * len(reference_words)
    results_decode = [[]] * len(reference_words)
    
    for word in reference_words:
        target_phonemes.append(df_targets[df_targets['Target'] == word]['Target_IPA'].values[0])
        error_list.append(df_targets[df_targets['Target'] == word]['Errors_Words'].values[0])
    
    # Segment the audio file into speech segments and return the logits matrix and character offsets for each segment
    _, logits_segments, char_offsets_segments = run_models(audio, target_duration_segment, min_duration_segment, 
                                                           overlap, reference_text, processor_phoneme, model_phoneme, 
                                                           processor_asr, model_asr, segmenter)
    
    # Process the logits matrix for each segment
    for logits_segment, char_offsets_segment in zip(logits_segments, char_offsets_segments): 
        # Split the logits at the first space character and return the sub-logits matrix, target phonemes, and error list
        sub_logits_segments = split_logit_matrix(logits_segment, char_offsets_segment)
        
        # Decode the most probable phonetic predictions for each segment
        all_decoded_predictions = []
        for sub_logits_segment in sub_logits_segments: ### process each subsegment of each segment in parallel
            decoded_predictions = multiple_predictions(sub_logits_segment, processor_phoneme, threshold_diff, 
                                                       max_pred_per_word, k, pad_token_id, space_token_id)
            all_decoded_predictions.append(decoded_predictions)
        
        # Binary classification of the reference text
        correct, _, results_decode = decode_phonetic_prediction_concat_error(all_decoded_predictions, target_phonemes,
                                                                              error_list, correct, results_decode) 
    
    return correct, results_decode


def process_targets(target_excel_path, target_sheet_name):
    """
    Process the target data from an Excel file.

    Args:
        target_excel_path (str): The file path of the Excel file containing the target data.
        target_sheet_name (str or int): The name or index of the sheet in the Excel file to read.

    Returns:
        pandas.DataFrame: The processed target data as a DataFrame.
    """
    df_targets = pd.read_excel(target_excel_path, sheet_name=target_sheet_name or 0)
    
    string_columns = ['Target_IPA', 'Asr_IPA_Other', 'Errors_Letter_End', 'Errors_Letter_Begining']
    for col in string_columns:
        df_targets[col] = df_targets[col].apply(lambda x: x.split() if isinstance(x, str) else [])
    
    df_targets['Target_IPA'] = df_targets.apply(lambda x: x['Target_IPA'] + x['Asr_IPA_Other'], axis=1)
    
    df_targets['Errors_Letter_End'] = df_targets.apply(
        lambda x: [word + letter for word in x['Target_IPA'] for letter in x['Errors_Letter_End']], axis=1)
    df_targets['Errors_Letter_Begining'] = df_targets.apply(
        lambda x: [letter + word for word in x['Target_IPA'] for letter in x['Errors_Letter_Begining']], axis=1)
    df_targets['Errors_Tilde'] = df_targets['Target_IPA'].apply(lambda x: [word + '\u0303' for word in x])
    
    df_targets['Errors_Words'] = df_targets.apply(
        lambda x: x['Errors_Letter_End'] + x['Errors_Letter_Begining'] + x['Errors_Tilde'], axis=1)
    
    df_targets['Errors_Words'] = df_targets['Errors_Words'].apply(lambda x: [i for i in x if i])
    df_targets['Target_IPA'] = df_targets['Target_IPA'].apply(lambda x: [i for i in x if i])
    
    return df_targets



##########################################################################################################

"""
Endpoint for segmenting audio and generating transcription and alignment results.

Returns:
    A JSON response containing the transcription and alignment results.

Raises:
    AssertionError: If the segmentation method is not supported.
    Exception: If there is an error during segmentation.

"""
@app.route('/asr_pipeline', methods=['POST']) 
def asr_pipeline():
    global sem
        
    sem.acquire()
    try:
        # Get data from request and load the audio file
        data = request.get_json()
        if not data or 'reference_text' not in data or 'audio' not in data:
            raise ValueError("Missing required data in request")

        reference_text = data['reference_text']
        audio_base64 = data['audio']
        audio_bytes = base64.b64decode(audio_base64)
        #load the audio file
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)

        # Run the pipeline on the audio file
        correct, results_decode = pipeline_asr_function(
            audio_file=audio,
            reference_text=reference_text,
            model_asr=model_asr_segmenter,
            processor_asr=processor_asr_segmenter,
            model_phoneme=model_phonemizer,
            processor_phoneme=processor_phonemizer,
            df_targets=df_targets,
            target_duration_segment=args.target_duration_segment,
            min_duration_segment=args.min_duration_segment,
            threshold_diff=args.threshold_diff,
            max_pred_per_word=args.max_pred_per_word,
            k=args.k,
            overlap=args.overlap,
            segmenter=args.segmenter,
            pad_token_id=pad_token_id,
            space_token_id=space_token_id
        )

        # Response to be sent back
        response = {
            'correct': correct,
            'results_decode': results_decode,
        }

    except ValueError as ve:
        app.logger.error(f"ValueError in segment route: {str(ve)}")
        sem.release()
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        app.logger.error(f"Unexpected error in segment route: {str(e)}")
        sem.release()
        return jsonify({'error': "An unexpected error occurred"}), 500

    sem.release()
    return jsonify(response), 201


    
##########################################################################################################

if __name__ == '__main__':
    ####### Parse the arguments of the pipeline #######
    parser = argparse.ArgumentParser()
    # Arguments for the Flask app
    parser.add_argument("--flask_port", help="The port of the flask app.", default=8070, type=int)
    # Arguments for the ASR models
    parser.add_argument("--id_asr_segmenter", type = str, default = "Dandan0K/Intervention-xls-FR-no-LM")
    parser.add_argument("--id_asr_phonemizer", type = str, default = "Cnam-LMSSC/wav2vec2-french-phonemizer")
    parser.add_argument("--segmenter", type = bool, default = True)
    # Arguments for loading the target excel file
    parser.add_argument("--target_excel_path" ,type = str, default = 'Targets_IPA.xlsx')
    parser.add_argument("--target_sheet_name", type = str, default = 0)
    # Arguments for the segmentation
    parser.add_argument("--target_duration_segment", type = float, default = 4)
    parser.add_argument("--target_duration_segment", type = float, default = 4)
    parser.add_argument("--min_duration_segment", type = float, default = 3)
    parser.add_argument("--overlap", type = bool, default = False)
    # Arguments for the decoder
    parser.add_argument("--threshold_diff", type = float, default = 2)
    parser.add_argument("--max_pred_per_word", type = int, default = 25)
    parser.add_argument("--top_k_is_all", type = bool, default = True)
    parser.add_argument("--k", type = int, default = 20)

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the target excel file
    df_targets = process_targets(args.target_excel_path, args.target_sheet_name)

    # Load the ASR segmenter processor and model (WithLM to use with hotwords)
    model_asr_segmenter =  Wav2Vec2ForCTC.from_pretrained(args.id_asr_segmenter).to(device)
    processor_asr_segmenter = Wav2Vec2ProcessorWithLM.from_pretrained(args.id_asr_segmenter)

    # Load the phonemizer processor and model
    model_phonemizer = Wav2Vec2ForCTC.from_pretrained(args.id_asr_phonemizer).to(device)
    processor_phonemizer = Wav2Vec2Processor.from_pretrained(args.id_asr_phonemizer)
 
    # If top_k_is_all is True, set k to the length of the vocab
    if args.top_k_is_all:
        args.k = len(processor_phonemizer.tokenizer.get_vocab())

    # Get the pad and space token id
    pad_token_id = processor_phonemizer.tokenizer.pad_token_id
    space_token_id = processor_phonemizer.tokenizer.convert_tokens_to_ids('|')
    
    sem = threading.Semaphore()    
    print("Waiting for requests...")
    app.run(host='0.0.0.0', port=args.flask_port, threaded = True )
