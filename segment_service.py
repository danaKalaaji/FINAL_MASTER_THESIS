import argparse
import json
import requests
from flask import Flask, jsonify, abort, make_response, request, Response
from flask_cors import CORS
from tqdm import tqdm
import os
import time
import threading
import base64
import io
from uuid import uuid4
#import whisperx
import gc 
import shutil
import time
#from ibm_watson import SpeechToTextV1
#from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import re
import numpy as np
import string
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2ProcessorWithLM
from itertools import groupby



# Make Flask application
app = Flask(__name__)
CORS(app)
# maintain the returned order of keys!
app.json.sort_keys = False
# regular expression to identify punctuation characters
punc_matcher = re.compile("[%s]"%(string.punctuation)) 


"""
Aligns two texts using the edit distance algorithm.

Args:
    text_a (str): The reference text.
    text_b (str): The model text.

Returns:
    list: A list of alignments between the two texts. Each alignment is represented as a tuple
            containing three elements: (word_a, word_b, operation).
"""
def align_texts(text_a, text_b):
    # Tokenize the texts into words: lower case + remove punctuation + split by space
    words_a = punc_matcher.sub(" ",text_a.lower()).split()
    words_b = punc_matcher.sub(" ",text_b.lower()).split()

    # Initialize the matrix with distances
    dp = np.zeros((len(words_a) + 1, len(words_b) + 1), dtype=int)
    
    # Initialize the first column and first row of the matrix
    for i in range(len(words_a) + 1):
        dp[i][0] = i
    for j in range(len(words_b) + 1):
        dp[0][j] = j

    # Compute the edit distance matrix
    for i in range(1, len(words_a) + 1):
        for j in range(1, len(words_b) + 1):
            cost = 0 if words_a[i - 1] == words_b[j - 1] else 1     
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # Deletion
                dp[i][j - 1] + 1,        # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            ) 

    # Trace back from dp[len(words_a)][len(words_b)] to dp[0][0] to find the alignment
    alignments = []
    i, j = len(words_a), len(words_b)
    while i > 0 and j > 0:
        if words_a[i - 1] == words_b[j - 1]:
            alignments.append((words_a[i - 1], words_b[j - 1], 'match'))
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i - 1][j - 1] + 1:
            alignments.append((words_a[i - 1], words_b[j - 1], 'substitution'))
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i][j - 1] + 1:
            alignments.append((None, words_b[j - 1], 'insertion'))
            j -= 1
        else:
            alignments.append((words_a[i - 1], None, 'deletion'))
            i -= 1

    # Capture any remaining words from text_a or text_b
    while i > 0:
        alignments.append((words_a[i - 1], None, 'deletion'))
        i -= 1
    while j > 0:
        alignments.append((None, words_b[j - 1], 'insertion'))
        j -= 1

    # The alignment list is built backwards, so reverse it
    alignments.reverse()

    return alignments

"""
Segments the audio file at the given path using the WhisperX ASR model.

Args:
    audio_path (str): The path to the audio file.
    language (str): The language code.

Returns:
    list: A list of segments representing the segmented audio.
"""
def whisperx_segment(audio_path, language, args, model, align_model, align_metadata):    
    if language not in align_model:
        model_a, metadata = whisperx.load_align_model(language_code=language, device=args.device)
        align_model[language] = model_a
        align_metadata[language] = metadata
        
    audio = whisperx.load_audio(audio_path)
    trans_result = model.transcribe(audio, batch_size=args.batch_size, language=language)
        
    seg_result = whisperx.align(trans_result["segments"], align_model[language], align_metadata[language], audio, args.device, return_char_alignments=False)["segments"]
    return seg_result
"""
Segment the audio file using IBM Watson Speech to Text service.

Args:
    audio_path (str): The path to the audio file.
    language (str): The language code.

Returns:
    list: A list of segments representing the segmented audio.
"""
def ibm_watson_segment(audio_path, language, ibm_asr_dict, speech_to_text): 
    
    base_model = ibm_asr_dict[language]["base_model"]
    model_id = ibm_asr_dict[language]["textReading"]["model_id"]
    with open(audio_path,'rb') as audio:
        recognition_job = speech_to_text.create_job(
                    audio,
                    model=base_model,
                    language_customization_id=model_id,
                    customization_weight=1,
                    speech_detector_sensitivity=0.6,
                    background_audio_suppression=0.1,
                    content_type='audio/wav',
                    timestamps=True,
                    word_confidence=True
                ).get_result()
    
    job_id = recognition_job["id"]
    tic = time.time()
    while True:
        recognition_job = speech_to_text.check_job(job_id).get_result()
        status = recognition_job["status"]
        if status == "completed":
            break
        elif status == "waiting":
            time.sleep(2.5)
            tac = time.time()
            if tac - tic > 60:
                print("Timeout!")
                break
        else:
            print("Job with ID "+job+" failed.")  
            break
            
    res = recognition_job
    words = []
    for word_time_stamps, word_confidence in map(list,zip(res["results"][0]["results"][0]["alternatives"][0]["timestamps"], res["results"][0]["results"][0]["alternatives"][0]["word_confidence"])):
        if word_time_stamps[0] == word_confidence[0]:
            word_time_stamps.append(word_confidence[1])
        else:
            word_time_stamps.append(0.0)
        words.append( { 'word': word_time_stamps[0], 'start': word_time_stamps[1], 'end': word_time_stamps[2], 'score': word_time_stamps[3] } )

    text = " ".join([w["word"] for w in words]) 
    start = words[0]["start"]
    end = words[-1]["end"]

    seg_result = [
        {
            "start":start,
            "end":end,
            "text":text,
            "words":words
        }
    ]
    return seg_result  

def wav2vec2_segment(audio_path, processor, model, with_LM=False):
    audio_segment, sr = librosa.load(audio_path, sr=16000)
    start_time = 0
    seg_result = []
   
    inputs = processor(audio_segment, sampling_rate=sr, return_tensors="pt", padding=True)
    #inputs = inputs.to('cuda')

    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    if not with_LM:
        transcription = processor.decode(predicted_ids[0]).lower()
        print(transcription)
    else:
        transcription = processor.batch_decode(logits.numpy()).text
        print(transcription)
        transcription = transcription[0].lower()

    words = [w for w in transcription.split(' ') if len(w) > 0]
    predicted_ids = predicted_ids[0].tolist()
    segment_duration_sec = inputs.input_values.shape[1] / sr
    """
    # create a list of tuples where each tuple contains the id of a word and the time in seconds when it was recognized
    ids_w_time = [(i / len(predicted_ids) * segment_duration_sec, _id) for i, _id in enumerate(predicted_ids)]
    # remove entries which are just "padding" (i.e. no characters are recognized)
    ids_w_time = [i for i in ids_w_time if i[1] != processor.tokenizer.pad_token_id]
    # now split the ids into groups of ids where each group represents a word
    split_ids_w_time = [list(group) for k, group
                        in groupby(ids_w_time, lambda x: x[1] == processor.tokenizer.word_delimiter_token_id)
                        if not k]

    assert len(split_ids_w_time) == len(words)  # make sure that there are the same number of id-groups as words. Otherwise something is wrong
    if len(split_ids_w_time) > 0:
        word_start_times = []
        word_end_times = []
        for cur_ids_w_time, cur_word in zip(split_ids_w_time, words):
            _times = [_time for _time, _id in cur_ids_w_time]
            word_start_times.append(min(_times))
            word_end_times.append(max(_times))
    
        seg_result.append({
            'start': start_time,
            'end': end_time,
            'text': ' '.join(words),
            'words': []
        })
        for word, start, end in zip(words, word_start_times, word_end_times):
            seg_result[-1]['words'].append({
                'word': word,
                'start': start,
                'end': end,
            })
        """
    seg_result.append({
    'start': start_time,
    'end': end_time,
    'text': ' '.join(words),
    'words': []
        })
    for word in words:
        seg_result[-1]['words'].append({
            'word': word,
            'start': 1,
            'end': 2,
        })
    return seg_result

"""
Converts the given list of results into a transcript dictionary.

Args:
    res (list): A list of dictionary items, each representing a segment of tue transcript.

Returns:
    dict: A dictionary containing the start time, end time, text, and words of the transcript.
"""
def get_transcript(res):
    start = 0
    end = 0
    text = []
    words = []
    for count, item in enumerate(res):
        if count == 0:
            start = item["start"]
        if count == len(res) - 1:
            end = item["end"]
        text.append( item["text"] )
        words += item["words"]
    return {
        "start":start,
        "end":end,
        "text":" ".join(text),
        "words":words
    }


"""
Endpoint for segmenting audio and generating transcription and alignment results.

Returns:
    A JSON response containing the transcription and alignment results.

Raises:
    AssertionError: If the segmentation method is not supported.
    Exception: If there is an error during segmentation.

"""
@app.route('/segment', methods=['POST']) 
def segment():
    global args, sem, models, processors_LM, processors_noLM
        
    sem.acquire()
    try:
        request_info = request.json
        audio_file_base64_string = request_info["audio_file_base64_string"]
        reference_text = request_info.get("reference_text", None)

        audio_save_folder = "temp_data/" + str(uuid4()) + "/"
        os.makedirs( audio_save_folder )

        with open( audio_save_folder + "audio.wav", "wb" ) as f:
            f.write( base64.b64decode(audio_file_base64_string) )
        audio_path = audio_save_folder + "audio.wav"
        language = request_info["language"].lower()
        method = request_info.get("method").lower()
        with_LM = request_info.get("with_LM", False)
        
        if method == "whisperx":
            model = whisperx.load_model(args.model_path, args.device, compute_type=args.compute_type)
            model_ID_list = request_info.get("model_ID_list", None)
            align_model = {}
            align_metadata = {}
            if model_ID_list is  None:
                for lan in ["it", "fr", "de"]:
                    model_a, metadata = whisperx.load_align_model(language_code = lan, device = args.device)
                    align_model[lan] = model_a
                    align_metadata[lan] = metadata
            else:
                for (lan, model_name) in zip(["it", "fr", "de"], model_ID_list):
                    model_a, metadata = whisperx.load_align_model(language_code = lan, device = args.device, model_name = model_name)
                    align_model[lan] = model_a
                    align_metadata[lan] = metadata

            #new_asr_options = model.options._asdict()
            #new_asr_options["initial_prompt"] = reference_text
            #new_options = faster_whisper.transcribe.TranscriptionOptions(**new_asr_options)
           # model.options = new_options
            seg_result = whisperx_segment( audio_path, language, args, model, align_model, align_metadata )

        elif re.sub("[^a-z]", "", method) == "ibmwatson":
            # Authentication of the IBM Watson service
            authenticator = IAMAuthenticator('4dTF6yq0l6JPOOjsngyOMh8iVbN3O_oKGIk_3oGPpAUr')
            speech_to_text = SpeechToTextV1(
                authenticator=authenticator
            )
            speech_to_text.set_service_url('https://api.eu-gb.speech-to-text.watson.cloud.ibm.com/instances/4c033d1b-cdee-4dfa-a26b-dc535fb10d72')
            ibm_asr_dict = {"fr":
                    {"base_model": "fr-FR_Multimedia",
                    "textReading":{"model_id": "f169d7db-0bd6-4f39-83c6-4fd2c39ea567",
                                    "path_audio": "recordings/fr/textReading/"},
                    },
                    "it":
                    {"base_model": "it-IT_Multimedia",
                    "textReading":{"model_id": "f1bcf049-a78a-4f5f-aec6-8e1978d4988e",
                                    "path_audio": "recordings/it/textReading/"},
                    },
                    "de":
                    {"base_model": "de-DE_Multimedia",
                    "textReading":{"model_id": "24c03ebd-c852-4b4a-8a69-197ee517ca8a",
                                    "path_audio": ""},
                    }
                }
        
            seg_result = ibm_watson_segment( audio_path, language,ibm_asr_dict, speech_to_text )

        elif method == "wav2vec2":
            model_ID_list = request_info.get("model_ID_list", None)
            if model_ID_list is None:
                raise Exception("MODEL_ID_list is not provided!")
        # Load the processor and model, index 0 is french, 1 is italian, 
            if with_LM:
                processors = processors_LM
            else:
                processors = processors_noLM

            if language == "fr":
                seg_result = wav2vec2_segment(audio_path, processors[0], models[0], with_LM=with_LM)
            elif language == "it":
                seg_result = wav2vec2_segment(audio_path, processors[1], models[1], with_LM=with_LM)
            else:
                raise ValueError("Invalid language. Please choose between 'fr' and 'it'.")
        else:
            assert False, "Unsupported segmentation method!"
        
        ## This is okay when the app is working in the single-thread mode
        shutil.rmtree( "temp_data/" )
    except Exception as e:
        # show the error 
        print(f"Error occurred during segmentation: {str(e)}")
        seg_result = []
    sem.release()

    transcript_result = get_transcript(seg_result)
    if reference_text is None:
        alignment_result = None
    else:
        alignment_result = align_texts(reference_text, transcript_result["text"])
    
    return jsonify({
        "transcription":transcript_result,
        "alignment":alignment_result
    }), 201
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--flask_port", help="The port of the flask app.", default=8070, type=int)
    parser.add_argument("--model_path", default = "large-v2")
    #parser.add_argument("--device", help="cpu or cuda", default = "cuda")
    parser.add_argument("--device", help="cpu or cuda", default = "cpu")

    parser.add_argument("--compute_type", help="int8/float16/float32", default = "int8")
    parser.add_argument("--batch_size", type = int, default = 4 )
    args = parser.parse_args()
    model_ID_list = ["Dandan0K/xls-r-1-Ref_french", 
                                        "Dandan0K/xls-r-1-Ref_italian"]
    processors_LM = [Wav2Vec2ProcessorWithLM.from_pretrained(id) for id in model_ID_list]
    processors_noLM =[Wav2Vec2Processor.from_pretrained(id) for id in model_ID_list] 
    models = [Wav2Vec2ForCTC.from_pretrained(id) for id in model_ID_list]
    for model in models:
        model.to(args.device)
    sem = threading.Semaphore()    
    print("Waiting for requests...")

    app.run(host='0.0.0.0', port=args.flask_port, threaded = True )
