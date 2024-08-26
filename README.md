# Installation

```bash
conda create --name whisperx python=3.10 -y
conda activate whisperx
conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install git+https://github.com/m-bain/whisperx.git
pip install -r requirements.txt
```
Install ffmpeg (https://github.com/openai/whisper#setup)  
- Linux:
```bash
sudo apt update && sudo apt install ffmpeg
```
- Windows:
1) Install chocolatey -> follow instruction in (https://chocolatey.org/install), run commands in Windows PowerShell
2) ```bash
    choco install ffmpeg
    ```
3) Check if ffmpeg installed and what version:
```bash
    ffmpeg.exe -version
```  
Install cudnn
```bash
conda install -c pypi cudnn -y
```
Install cudnn
```bash
conda install -c pypi cudnn -y
```

# Start service
The service will listen to port 8070
```bash
python segment-service.py
```

# Call the API

## Define the caller function


```python
import requests,json,base64
import pandas as pd
import os
from glob import glob

## define a function for segmentation
def call_segment_service( service_address, 
                          audio_file_path,
                          reference_text,
                          language,
                          method
                        ):
    audio_file_base64_string = base64.b64encode( open(audio_file_path, 'rb').read()).decode('ASCII')
    response = requests.post( service_address,
                              data = json.dumps( {
                                  "audio_file_base64_string":audio_file_base64_string,
                                  "reference_text":reference_text,
                                  "language":language,
                                  "method":method
                              } ),
                              headers = {"Content-Type": "application/json"}
                            )
    return response.json()
```

## Segment Child Speech using WhisperX

## Get the corresponding reference text of each audio file


```python
reading_material={
    "it":{
        "A":[
            "I draghi sono di solito giganteschi e pericolosi. Hanno squame, artigli e ali. Sembrano un incrocio tra un serpente, un pesce e un uccello. Alcuni draghi sputano fuoco. Altri hanno molte teste.",
            "I draghi depongono delle uova. Durante la schiusa, i piccoli draghi emergono dalle uova. I giovani draghi crescono molto lentamente. I draghi adulti possono vivere molto a lungo.",
            "I draghi esistono solo nei racconti. Ci sono racconti sui draghi in tantissimi paesi del mondo: in Cina, in Persia, in Grecia e in altri paesi europei. Benché nella maggior parte dei casi i draghi siano malvagi, mangiatori di uomini o guardiani di un tesoro, in Cina sono considerati come portafortuna."
        ],
        "B":[
            "Gli unicorni sono bianchi. Sono animali selvatici che assomigliano molto ai cavalli. Gli unicorni hanno un corno lungo e appuntito. Questo corno è contorto, un po' come una spirale. Il corno cresce in mezzo alla fronte.",
            "Gli unicorni vivono molto a lungo. Le leggende dicono che possono vivere anche secoli. I piccoli unicorni sono chiamati puledri. Le loro corna rimangono nascoste nella criniera fino all'età adulta.",
            "Gli unicorni sono un esempio perfetto di creature mitologiche. Si raccontavano storie di unicorni già a Roma e nella Grecia antica, ma anche in altri paesi europei. Questo animale mitologico è simbolo di purezza, perché il suo corno è un antidoto. È indomabile e molto difficile da catturare."
        ],
        "C":[
            "I nani sono in genere piccoli e carini. I nani assomigliano agli uomini. Hanno lunghe barbe bianche e portano un berretto a punta. Vivono spesso in gruppo nella stessa grotta. Sono in genere amichevoli.",
            "I nani lavorano sottoterra nelle miniere. Estraggono pietre preziose e oro dal sottosuolo. Sono incredibili gioiellieri. I loro gioielli sono splendidi e si vendono a caro prezzo.",
            "I nani compaiono spesso nelle fiabe. La maggior parte delle storie di nani proviene dalla Scandinavia, ma anche dalla Germania e da altri paesi europei. Sebbene i nani siano spesso di grande aiuto per gli esseri umani (e in questo caso sono chiamati spiriti buoni del focolare), ci sono tra loro anche dei guerrieri."
        ]
    },
    "fr":{
        "A":[
            "Les dragons sont souvent grands et dangereux. Les dragons ont des écailles, des griffes et des ailes. On dirait un croisement entre un serpent, un poisson et un oiseau. Certains dragons crachent du feu. D’autres ont plus d’une tête. ",
            "Les dragons pondent des oeufs. Lors de l'éclosion, les bébés dragons sortent des oeufs. Les jeunes dragons grandissent très lentement. Les dragons adultes peuvent alors vivre très vieux.",
            "Les dragons existent uniquement dans les contes. Il y a des histoires de dragons partout dans le monde, on en raconte en Chine, en Iran, en Grèce, et aussi dans d’autres pays européens. Bien que la plupart des dragons soient maléfiques, dévorent les humains ou protègent de précieux trésors, ils sont considérés comme des porte-bonheurs en Chine."
        ],
        "B":[
            "Une licorne est blanche. C'est un animal sauvage qui ressemble énormément à un cheval. Les licornes ont une corne longue et pointue. Leur corne est torsadée, un peu comme une spirale. Elle pousse au milieu de leur front.",
            "Les licornes vivent très longtemps. Des légendes racontent qu’elles pourraient même vivre plusieurs siècles. Les bébés licornes sont appelés poulains. Leur corne reste cachée dans leur crinière jusqu’à l’âge adulte.",
            "La licorne est l’exemple parfait de la créature mythique. On racontait déjà des histoires de licornes à Rome et en Grèce antique, mais aussi dans d’autres pays européens. Cet animal mythique est un signe de pureté, parce que sa corne est un antidote. Il est indomptable et très compliqué à capturer."
        ],
        "C":[
            "Les nains sont généralement petits et mignons. Les nains ressemblent à des hommes. Les nains ont de longues barbes blanches et portent un bonnet pointu. Ils vivent souvent à plusieurs dans la même grotte. Ils sont généralement amicaux.",
            "Les nains travaillent sous terre dans les mines. Ils extraient des pierres précieuses et de l’or du sous-sol. Ils sont d’incroyables bijoutiers. Leurs bijoux sont magnifiques et se vendent à prix d’or.",
            "Les nains apparaissent fréquemment dans les contes de fées. La plupart des histoires de nains proviennent de Scandinavie, mais aussi d'Allemagne ou d'autres pays européens. Quoique les nains soient parfois d’une aide considérable pour les humains, et sont alors appelés bons génies du foyer, il y a néanmoins des guerriers parmi eux."
        ]
    }
}
```


```python
df = pd.read_csv("data/example/participants_extract.csv")
data_table = {"audio_path":[], "fname":[], "participant_id":[], "language":[],"form":[], "order":[], "reference_text":[]}
for path_name in glob("data/example/*/*.wav"):
    fname = os.path.basename( path_name )
    participant_id = int(fname.split("_")[0])
    lan = list(df[df["participant_id"] == participant_id]["language"])[0]
    form = list(df[df["participant_id"] == participant_id]["form"])[0]
    order = list(df[df["participant_id"] == participant_id]["order"])[0]

    if order == 0:
        print("skip ... likely invalid trial.")
        continue
    reference_text = reading_material[lan][form][order -1 ]

    data_table["audio_path"].append( path_name )
    data_table["fname"].append( fname )
    data_table["participant_id"].append( participant_id )
    data_table["language"].append( lan )
    data_table["form"].append( form )
    data_table["order"].append( order )
    data_table["reference_text"].append( reference_text )

data_table_df = pd.DataFrame( data_table )
data_table_df
```

    skip ... likely invalid trial.
    skip ... likely invalid trial.
    skip ... likely invalid trial.
    skip ... likely invalid trial.
    skip ... likely invalid trial.
    skip ... likely invalid trial.
    skip ... likely invalid trial.
    skip ... likely invalid trial.





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>audio_path</th>
      <th>fname</th>
      <th>participant_id</th>
      <th>language</th>
      <th>form</th>
      <th>order</th>
      <th>reference_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>data/example/french/3101_edugame2023_823f70075...</td>
      <td>3101_edugame2023_823f700759e54f17aae929dfce128...</td>
      <td>3101</td>
      <td>fr</td>
      <td>B</td>
      <td>2</td>
      <td>Les licornes vivent très longtemps. Des légend...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>data/example/french/3116_edugame2023_343062cee...</td>
      <td>3116_edugame2023_343062cee0ce4572b692dc1108d03...</td>
      <td>3116</td>
      <td>fr</td>
      <td>C</td>
      <td>2</td>
      <td>Les nains travaillent sous terre dans les mine...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>data/example/french/3117_edugame2023_a4257b0c1...</td>
      <td>3117_edugame2023_a4257b0c11684b929f531dc73b0c5...</td>
      <td>3117</td>
      <td>fr</td>
      <td>A</td>
      <td>3</td>
      <td>Les dragons existent uniquement dans les conte...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>data/example/french/3103_edugame2023_a8186e5c5...</td>
      <td>3103_edugame2023_a8186e5c579c451692f69a0ddbee3...</td>
      <td>3103</td>
      <td>fr</td>
      <td>B</td>
      <td>1</td>
      <td>Une licorne est blanche. C'est un animal sauva...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>data/example/french/3104_edugame2023_0a936d5f3...</td>
      <td>3104_edugame2023_0a936d5f3e25497fb00a4fcdbf72a...</td>
      <td>3104</td>
      <td>fr</td>
      <td>C</td>
      <td>1</td>
      <td>Les nains sont généralement petits et mignons....</td>
    </tr>
    <tr>
      <th>5</th>
      <td>data/example/french/3118_edugame2023_9ba06a94f...</td>
      <td>3118_edugame2023_9ba06a94f4a4407ea3b2dbdf2254f...</td>
      <td>3118</td>
      <td>fr</td>
      <td>B</td>
      <td>1</td>
      <td>Une licorne est blanche. C'est un animal sauva...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>data/example/french/3110_edugame2023_01e806baa...</td>
      <td>3110_edugame2023_01e806baa12f4df891326e4acade6...</td>
      <td>3110</td>
      <td>fr</td>
      <td>A</td>
      <td>1</td>
      <td>Les dragons sont souvent grands et dangereux. ...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>data/example/french/3111_edugame2023_090bbbdda...</td>
      <td>3111_edugame2023_090bbbdda6cc499d8f12cbec6256e...</td>
      <td>3111</td>
      <td>fr</td>
      <td>B</td>
      <td>3</td>
      <td>La licorne est l’exemple parfait de la créatur...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>data/example/french/3106_edugame2023_8e887e74d...</td>
      <td>3106_edugame2023_8e887e74d4414eecad4236774c62d...</td>
      <td>3106</td>
      <td>fr</td>
      <td>A</td>
      <td>3</td>
      <td>Les dragons existent uniquement dans les conte...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>data/example/french/3108_edugame2023_3c0a78208...</td>
      <td>3108_edugame2023_3c0a78208d0c416c852225bccf336...</td>
      <td>3108</td>
      <td>fr</td>
      <td>C</td>
      <td>2</td>
      <td>Les nains travaillent sous terre dans les mine...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>data/example/french/3109_edugame2023_00b0af188...</td>
      <td>3109_edugame2023_00b0af1883a14ceeb73a49704187e...</td>
      <td>3109</td>
      <td>fr</td>
      <td>C</td>
      <td>1</td>
      <td>Les nains sont généralement petits et mignons....</td>
    </tr>
    <tr>
      <th>11</th>
      <td>data/example/italian/120_edugame2023_56123f4b0...</td>
      <td>120_edugame2023_56123f4b0dac457a8d8384766495a9...</td>
      <td>120</td>
      <td>it</td>
      <td>A</td>
      <td>2</td>
      <td>I draghi depongono delle uova. Durante la schi...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>data/example/italian/102_edugame2023_67b3311a8...</td>
      <td>102_edugame2023_67b3311a82a84f49a9c76d23fb3850...</td>
      <td>102</td>
      <td>it</td>
      <td>C</td>
      <td>3</td>
      <td>I nani compaiono spesso nelle fiabe. La maggio...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>data/example/italian/119_edugame2023_63cc46f05...</td>
      <td>119_edugame2023_63cc46f057a8491dbdda9ca7bd749b...</td>
      <td>119</td>
      <td>it</td>
      <td>B</td>
      <td>2</td>
      <td>Gli unicorni vivono molto a lungo. Le leggende...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>data/example/italian/106_edugame2023_13ab8f60b...</td>
      <td>106_edugame2023_13ab8f60ba604cba899de645a5b013...</td>
      <td>106</td>
      <td>it</td>
      <td>A</td>
      <td>2</td>
      <td>I draghi depongono delle uova. Durante la schi...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>data/example/italian/123_edugame2023_d4ccb4a46...</td>
      <td>123_edugame2023_d4ccb4a46ed641dd8f4f6daf6f4761...</td>
      <td>123</td>
      <td>it</td>
      <td>B</td>
      <td>3</td>
      <td>Gli unicorni sono un esempio perfetto di creat...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>data/example/italian/101_edugame2023_f4f48951f...</td>
      <td>101_edugame2023_f4f48951fc4d47daa509dc81c04d79...</td>
      <td>101</td>
      <td>it</td>
      <td>B</td>
      <td>3</td>
      <td>Gli unicorni sono un esempio perfetto di creat...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>data/example/italian/116_edugame2023_fa1d0c9a6...</td>
      <td>116_edugame2023_fa1d0c9a6dda4752929f099179e816...</td>
      <td>116</td>
      <td>it</td>
      <td>C</td>
      <td>1</td>
      <td>I nani sono in genere piccoli e carini. I nani...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>data/example/italian/105_edugame2023_c94109dca...</td>
      <td>105_edugame2023_c94109dca75c4921a7203ec6316755...</td>
      <td>105</td>
      <td>it</td>
      <td>C</td>
      <td>3</td>
      <td>I nani compaiono spesso nelle fiabe. La maggio...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>data/example/italian/104_edugame2023_7f9e09461...</td>
      <td>104_edugame2023_7f9e0946108c4f93a79a729b12832f...</td>
      <td>104</td>
      <td>it</td>
      <td>A</td>
      <td>3</td>
      <td>I draghi esistono solo nei racconti. Ci sono r...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>data/example/italian/108_edugame2023_a1bead214...</td>
      <td>108_edugame2023_a1bead214f3c46dfbf1513875337b6...</td>
      <td>108</td>
      <td>it</td>
      <td>A</td>
      <td>3</td>
      <td>I draghi esistono solo nei racconti. Ci sono r...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>data/example/italian/121_edugame2023_ab2346aa3...</td>
      <td>121_edugame2023_ab2346aa3c7147f384d8c025709cac...</td>
      <td>121</td>
      <td>it</td>
      <td>A</td>
      <td>1</td>
      <td>I draghi sono di solito giganteschi e pericolo...</td>
    </tr>
  </tbody>
</table>
</div>



## Transcribe an audio, and align with the reference text

### check the french


```python
participant_id  = 3104
data_table_df.loc[ data_table_df["participant_id"] == participant_id ]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>audio_path</th>
      <th>fname</th>
      <th>participant_id</th>
      <th>language</th>
      <th>form</th>
      <th>order</th>
      <th>reference_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>data/example/french/3104_edugame2023_0a936d5f3...</td>
      <td>3104_edugame2023_0a936d5f3e25497fb00a4fcdbf72a...</td>
      <td>3104</td>
      <td>fr</td>
      <td>C</td>
      <td>1</td>
      <td>Les nains sont généralement petits et mignons....</td>
    </tr>
  </tbody>
</table>
</div>




```python
audio_path = list(data_table_df.loc[ data_table_df["participant_id"] == participant_id ]["audio_path"])[0]
reference_text = list(data_table_df.loc[ data_table_df["participant_id"] == participant_id ]["reference_text"])[0]
language = list(data_table_df.loc[ data_table_df["participant_id"] == participant_id ]["language"])[0]
audio_path, reference_text, language
```




    ('data/example/french/3104_edugame2023_0a936d5f3e25497fb00a4fcdbf72aac6_9cbecd83e198456db059ff879bc134a8.wav',
     'Les nains sont généralement petits et mignons. Les nains ressemblent à des hommes. Les nains ont de longues barbes blanches et portent un bonnet pointu. Ils vivent souvent à plusieurs dans la même grotte. Ils sont généralement amicaux.',
     'fr')




```python
call_segment_service( 
            service_address = "http://localhost:8070/segment",  ### This is the service address, default on 8070 port.
            audio_file_path = audio_path,  ### path to the audio file
            reference_text = reference_text,
            language = language,   ### "it" for "italian", "fr" for "frence", "de" for "german"
            method = "whisperx"  ### The name of the model, can be either "whisperx" or "ibm-watson"
)
```




    {'transcription': {'start': 0.009,
      'end': 90.66,
      'text': " Lorsque tu as terminé, tu peux appuyer en bas. Appuie bien fort. Les nains sont généralement petits et mignons. Les nains ressemblent à des hommes. Les nains ont de longues bras blancs et portent des bonnets pointus.  Ils vivent souvent en plusieurs dans la même grotte. Ils sont généralement amicaux. Les nains travaillent sous la terre dans les mines.  ils extraient des pierres précieuses et de l'or du sous-sol et ils sont d'incroyables bijoux et leurs bijoux sont magnifiques et ce  Vendôme, ce vendôme a pris d'or. Terminé.",
      'words': [{'word': 'Lorsque', 'start': 0.009, 'end': 0.529, 'score': 0.418},
       {'word': 'tu', 'start': 0.609, 'end': 0.789, 'score': 0.536},
       {'word': 'as', 'start': 0.849, 'end': 0.909, 'score': 0.204},
       {'word': 'terminé,', 'start': 0.929, 'end': 1.809, 'score': 0.503},
       {'word': 'tu', 'start': 1.91, 'end': 2.49, 'score': 0.868},
       {'word': 'peux', 'start': 2.51, 'end': 2.83, 'score': 0.526},
       {'word': 'appuyer', 'start': 2.85, 'end': 3.03, 'score': 0.204},
       {'word': 'en', 'start': 3.11, 'end': 4.15, 'score': 0.81},
       {'word': 'bas.', 'start': 4.17, 'end': 4.31, 'score': 0.569},
       {'word': 'Appuie', 'start': 4.79, 'end': 5.631, 'score': 0.52},
       {'word': 'bien', 'start': 5.671, 'end': 6.031, 'score': 0.611},
       {'word': 'fort.', 'start': 6.051, 'end': 6.331, 'score': 0.337},
       {'word': 'Les', 'start': 6.371, 'end': 6.711, 'score': 0.686},
       {'word': 'nains', 'start': 6.751, 'end': 7.011, 'score': 0.277},
       {'word': 'sont', 'start': 7.071, 'end': 7.271, 'score': 0.417},
       {'word': 'généralement', 'start': 7.311, 'end': 8.071, 'score': 0.321},
       {'word': 'petits', 'start': 8.111, 'end': 8.671, 'score': 0.91},
       {'word': 'et', 'start': 8.711, 'end': 9.011, 'score': 0.676},
       {'word': 'mignons.', 'start': 9.031, 'end': 10.272, 'score': 0.664},
       {'word': 'Les', 'start': 10.332, 'end': 10.732, 'score': 0.782},
       {'word': 'nains', 'start': 10.812, 'end': 11.872, 'score': 0.51},
       {'word': 'ressemblent', 'start': 11.912, 'end': 13.092, 'score': 0.771},
       {'word': 'à', 'start': 13.153, 'end': 13.993, 'score': 0.962},
       {'word': 'des', 'start': 14.033, 'end': 14.633, 'score': 0.91},
       {'word': 'hommes.', 'start': 14.693, 'end': 18.394, 'score': 0.736},
       {'word': 'Les', 'start': 18.434, 'end': 18.594, 'score': 0.68},
       {'word': 'nains', 'start': 18.654, 'end': 19.394, 'score': 0.57},
       {'word': 'ont', 'start': 19.454, 'end': 20.394, 'score': 0.518},
       {'word': 'de', 'start': 20.414, 'end': 20.775, 'score': 0.858},
       {'word': 'longues', 'start': 20.835, 'end': 22.015, 'score': 0.848},
       {'word': 'bras', 'start': 22.035, 'end': 22.815, 'score': 0.658},
       {'word': 'blancs', 'start': 22.855, 'end': 24.436, 'score': 0.749},
       {'word': 'et', 'start': 24.496, 'end': 25.196, 'score': 0.893},
       {'word': 'portent', 'start': 25.276, 'end': 26.656, 'score': 0.749},
       {'word': 'des', 'start': 26.696, 'end': 27.256, 'score': 0.905},
       {'word': 'bonnets', 'start': 27.296, 'end': 28.097, 'score': 0.706},
       {'word': 'pointus.', 'start': 28.117, 'end': 28.657, 'score': 0.659},
       {'word': 'Ils', 'start': 30.556, 'end': 31.696, 'score': 0.708},
       {'word': 'vivent', 'start': 31.736, 'end': 32.676, 'score': 0.514},
       {'word': 'souvent', 'start': 32.716, 'end': 34.297, 'score': 0.731},
       {'word': 'en', 'start': 34.357, 'end': 34.797, 'score': 0.714},
       {'word': 'plusieurs', 'start': 34.837, 'end': 37.177, 'score': 0.534},
       {'word': 'dans', 'start': 37.197, 'end': 37.697, 'score': 0.532},
       {'word': 'la', 'start': 37.737, 'end': 38.758, 'score': 0.732},
       {'word': 'même', 'start': 38.798, 'end': 41.118, 'score': 0.734},
       {'word': 'grotte.', 'start': 41.158, 'end': 42.238, 'score': 0.762},
       {'word': 'Ils', 'start': 42.318, 'end': 42.818, 'score': 0.737},
       {'word': 'sont', 'start': 42.838, 'end': 43.278, 'score': 0.558},
       {'word': 'généralement', 'start': 43.318, 'end': 46.119, 'score': 0.56},
       {'word': 'amicaux.', 'start': 46.199, 'end': 48.92, 'score': 0.606},
       {'word': 'Les', 'start': 49.28, 'end': 50.12, 'score': 0.769},
       {'word': 'nains', 'start': 50.38, 'end': 51.0, 'score': 0.411},
       {'word': 'travaillent', 'start': 51.06, 'end': 52.12, 'score': 0.778},
       {'word': 'sous', 'start': 52.16, 'end': 52.78, 'score': 0.756},
       {'word': 'la', 'start': 52.84, 'end': 53.38, 'score': 0.712},
       {'word': 'terre', 'start': 53.4, 'end': 54.221, 'score': 0.605},
       {'word': 'dans', 'start': 54.241, 'end': 55.281, 'score': 0.516},
       {'word': 'les', 'start': 55.321, 'end': 55.541, 'score': 0.534},
       {'word': 'mines.', 'start': 55.561, 'end': 55.681, 'score': 0.126},
       {'word': 'ils', 'start': 57.216, 'end': 57.716, 'score': 0.653},
       {'word': 'extraient', 'start': 57.796, 'end': 60.357, 'score': 0.424},
       {'word': 'des', 'start': 60.397, 'end': 61.018, 'score': 0.74},
       {'word': 'pierres', 'start': 61.058, 'end': 64.019, 'score': 0.382},
       {'word': 'précieuses', 'start': 64.159, 'end': 65.9, 'score': 0.538},
       {'word': 'et', 'start': 65.96, 'end': 66.8, 'score': 0.824},
       {'word': 'de', 'start': 66.82, 'end': 67.16, 'score': 0.791},
       {'word': "l'or", 'start': 67.38, 'end': 68.08, 'score': 0.789},
       {'word': 'du', 'start': 68.12, 'end': 68.561, 'score': 0.84},
       {'word': 'sous-sol', 'start': 68.601, 'end': 70.421, 'score': 0.702},
       {'word': 'et', 'start': 70.481, 'end': 70.761, 'score': 0.731},
       {'word': 'ils', 'start': 70.862, 'end': 71.422, 'score': 0.826},
       {'word': 'sont', 'start': 71.462, 'end': 72.002, 'score': 0.574},
       {'word': "d'incroyables", 'start': 72.062, 'end': 75.183, 'score': 0.514},
       {'word': 'bijoux', 'start': 75.203, 'end': 77.664, 'score': 0.534},
       {'word': 'et', 'start': 77.964, 'end': 78.565, 'score': 0.788},
       {'word': 'leurs', 'start': 78.605, 'end': 79.365, 'score': 0.604},
       {'word': 'bijoux', 'start': 79.385, 'end': 80.165, 'score': 0.519},
       {'word': 'sont', 'start': 80.205, 'end': 80.766, 'score': 0.654},
       {'word': 'magnifiques', 'start': 80.806, 'end': 85.727, 'score': 0.726},
       {'word': 'et', 'start': 85.808, 'end': 86.608, 'score': 0.794},
       {'word': 'ce', 'start': 86.668, 'end': 86.748, 'score': 0.467},
       {'word': 'Vendôme,', 'start': 87.517, 'end': 87.717, 'score': 0.144},
       {'word': 'ce', 'start': 87.777, 'end': 87.857, 'score': 0.362},
       {'word': 'vendôme', 'start': 87.877, 'end': 88.218, 'score': 0.391},
       {'word': 'a', 'start': 88.238, 'end': 88.258, 'score': 0.003},
       {'word': 'pris', 'start': 88.278, 'end': 89.159, 'score': 0.459},
       {'word': "d'or.", 'start': 89.199, 'end': 89.279, 'score': 0.283},
       {'word': 'Terminé.', 'start': 89.339, 'end': 90.66, 'score': 0.582}]},
     'alignment': [[None, 'lorsque', 'insertion'],
      [None, 'tu', 'insertion'],
      [None, 'as', 'insertion'],
      [None, 'terminé', 'insertion'],
      [None, 'tu', 'insertion'],
      [None, 'peux', 'insertion'],
      [None, 'appuyer', 'insertion'],
      [None, 'en', 'insertion'],
      [None, 'bas', 'insertion'],
      [None, 'appuie', 'insertion'],
      [None, 'bien', 'insertion'],
      [None, 'fort', 'insertion'],
      ['les', 'les', 'match'],
      ['nains', 'nains', 'match'],
      ['sont', 'sont', 'match'],
      ['généralement', 'généralement', 'match'],
      ['petits', 'petits', 'match'],
      ['et', 'et', 'match'],
      ['mignons', 'mignons', 'match'],
      ['les', 'les', 'match'],
      ['nains', 'nains', 'match'],
      ['ressemblent', 'ressemblent', 'match'],
      ['à', 'à', 'match'],
      ['des', 'des', 'match'],
      ['hommes', 'hommes', 'match'],
      ['les', 'les', 'match'],
      ['nains', 'nains', 'match'],
      ['ont', 'ont', 'match'],
      ['de', 'de', 'match'],
      ['longues', 'longues', 'match'],
      ['barbes', 'bras', 'substitution'],
      ['blanches', 'blancs', 'substitution'],
      ['et', 'et', 'match'],
      ['portent', 'portent', 'match'],
      ['un', 'des', 'substitution'],
      ['bonnet', 'bonnets', 'substitution'],
      ['pointu', 'pointus', 'substitution'],
      ['ils', 'ils', 'match'],
      ['vivent', 'vivent', 'match'],
      ['souvent', 'souvent', 'match'],
      ['à', 'en', 'substitution'],
      ['plusieurs', 'plusieurs', 'match'],
      ['dans', 'dans', 'match'],
      ['la', 'la', 'match'],
      ['même', 'même', 'match'],
      ['grotte', 'grotte', 'match'],
      ['ils', 'ils', 'match'],
      ['sont', 'sont', 'match'],
      ['généralement', 'généralement', 'match'],
      ['amicaux', 'amicaux', 'match'],
      [None, 'les', 'insertion'],
      [None, 'nains', 'insertion'],
      [None, 'travaillent', 'insertion'],
      [None, 'sous', 'insertion'],
      [None, 'la', 'insertion'],
      [None, 'terre', 'insertion'],
      [None, 'dans', 'insertion'],
      [None, 'les', 'insertion'],
      [None, 'mines', 'insertion'],
      [None, 'ils', 'insertion'],
      [None, 'extraient', 'insertion'],
      [None, 'des', 'insertion'],
      [None, 'pierres', 'insertion'],
      [None, 'précieuses', 'insertion'],
      [None, 'et', 'insertion'],
      [None, 'de', 'insertion'],
      [None, 'l', 'insertion'],
      [None, 'or', 'insertion'],
      [None, 'du', 'insertion'],
      [None, 'sous', 'insertion'],
      [None, 'sol', 'insertion'],
      [None, 'et', 'insertion'],
      [None, 'ils', 'insertion'],
      [None, 'sont', 'insertion'],
      [None, 'd', 'insertion'],
      [None, 'incroyables', 'insertion'],
      [None, 'bijoux', 'insertion'],
      [None, 'et', 'insertion'],
      [None, 'leurs', 'insertion'],
      [None, 'bijoux', 'insertion'],
      [None, 'sont', 'insertion'],
      [None, 'magnifiques', 'insertion'],
      [None, 'et', 'insertion'],
      [None, 'ce', 'insertion'],
      [None, 'vendôme', 'insertion'],
      [None, 'ce', 'insertion'],
      [None, 'vendôme', 'insertion'],
      [None, 'a', 'insertion'],
      [None, 'pris', 'insertion'],
      [None, 'd', 'insertion'],
      [None, 'or', 'insertion'],
      [None, 'terminé', 'insertion']]}



### Check the Italian


```python
participant_id  = 102
data_table_df.loc[ data_table_df["participant_id"] == participant_id ]

audio_path = list(data_table_df.loc[ data_table_df["participant_id"] == participant_id ]["audio_path"])[0]
reference_text = list(data_table_df.loc[ data_table_df["participant_id"] == participant_id ]["reference_text"])[0]
language = list(data_table_df.loc[ data_table_df["participant_id"] == participant_id ]["language"])[0]
audio_path, reference_text, language
```




    ('data/example/italian/102_edugame2023_67b3311a82a84f49a9c76d23fb385084_2e5d8a11f48f4c3eae6dff081ae5059f.wav',
     'I nani compaiono spesso nelle fiabe. La maggior parte delle storie di nani proviene dalla Scandinavia, ma anche dalla Germania e da altri paesi europei. Sebbene i nani siano spesso di grande aiuto per gli esseri umani (e in questo caso sono chiamati spiriti buoni del focolare), ci sono tra loro anche dei guerrieri.',
     'it')




```python
call_segment_service( 
            service_address = "http://localhost:8070/segment",  ### This is the service address, default on 8070 port.
            audio_file_path = audio_path,  ### path to the audio file
            reference_text = reference_text,
            language = language,   ### "it" for "italian", "fr" for "frence", "de" for "german"
            method = "whisperx"  ### The name of the model, can be either "whisperx" or "ibm-watson"
)
```




    {'transcription': {'start': 0.149,
      'end': 46.812,
      'text': ' i nani capiscono spesso nelle viabbe la maggior parte delle storie di nani proviene dalla scandinavia ma anche dalla germania e  da altri paesi europei, sebbene i nani siano spesso di grande aiuto per gli',
      'words': [{'word': 'i', 'start': 0.149, 'end': 0.429, 'score': 0.173},
       {'word': 'nani', 'start': 2.71, 'end': 3.01, 'score': 0.886},
       {'word': 'capiscono', 'start': 3.07, 'end': 3.911, 'score': 0.884},
       {'word': 'spesso', 'start': 4.071, 'end': 4.891, 'score': 0.939},
       {'word': 'nelle', 'start': 5.352, 'end': 5.812, 'score': 0.919},
       {'word': 'viabbe', 'start': 5.952, 'end': 6.532, 'score': 0.86},
       {'word': 'la', 'start': 6.612, 'end': 6.852, 'score': 0.821},
       {'word': 'maggior', 'start': 8.173, 'end': 8.893, 'score': 0.91},
       {'word': 'parte', 'start': 9.053, 'end': 9.554, 'score': 0.976},
       {'word': 'delle', 'start': 9.674, 'end': 10.254, 'score': 0.962},
       {'word': 'storie', 'start': 11.254, 'end': 11.995, 'score': 0.866},
       {'word': 'di', 'start': 12.195, 'end': 12.415, 'score': 0.965},
       {'word': 'nani', 'start': 13.055, 'end': 13.576, 'score': 0.816},
       {'word': 'proviene', 'start': 14.016, 'end': 14.896, 'score': 0.876},
       {'word': 'dalla', 'start': 15.296, 'end': 16.157, 'score': 0.95},
       {'word': 'scandinavia', 'start': 20.679, 'end': 21.499, 'score': 0.915},
       {'word': 'ma', 'start': 21.88, 'end': 22.02, 'score': 0.962},
       {'word': 'anche', 'start': 22.38, 'end': 22.82, 'score': 0.859},
       {'word': 'dalla', 'start': 23.58, 'end': 24.081, 'score': 0.927},
       {'word': 'germania', 'start': 25.181, 'end': 26.102, 'score': 0.875},
       {'word': 'e', 'start': 26.122, 'end': 26.142, 'score': 0.005},
       {'word': 'da', 'start': 29.483, 'end': 29.663, 'score': 0.788},
       {'word': 'altri', 'start': 30.143, 'end': 30.664, 'score': 0.874},
       {'word': 'paesi', 'start': 30.884, 'end': 31.544, 'score': 0.933},
       {'word': 'europei,', 'start': 31.764, 'end': 32.505, 'score': 0.953},
       {'word': 'sebbene', 'start': 35.446, 'end': 36.227, 'score': 0.816},
       {'word': 'i', 'start': 37.107, 'end': 37.287, 'score': 0.919},
       {'word': 'nani', 'start': 38.108, 'end': 38.688, 'score': 0.814},
       {'word': 'siano', 'start': 40.249, 'end': 40.849, 'score': 0.865},
       {'word': 'spesso', 'start': 42.45, 'end': 43.05, 'score': 0.945},
       {'word': 'di', 'start': 43.15, 'end': 43.25, 'score': 0.95},
       {'word': 'grande', 'start': 43.35, 'end': 43.971, 'score': 0.956},
       {'word': 'aiuto', 'start': 45.091, 'end': 45.932, 'score': 0.93},
       {'word': 'per', 'start': 46.192, 'end': 46.492, 'score': 0.905},
       {'word': 'gli', 'start': 46.752, 'end': 46.812, 'score': 0.064}]},
     'alignment': [['i', 'i', 'match'],
      ['nani', 'nani', 'match'],
      ['compaiono', 'capiscono', 'substitution'],
      ['spesso', 'spesso', 'match'],
      ['nelle', 'nelle', 'match'],
      ['fiabe', 'viabbe', 'substitution'],
      ['la', 'la', 'match'],
      ['maggior', 'maggior', 'match'],
      ['parte', 'parte', 'match'],
      ['delle', 'delle', 'match'],
      ['storie', 'storie', 'match'],
      ['di', 'di', 'match'],
      ['nani', 'nani', 'match'],
      ['proviene', 'proviene', 'match'],
      ['dalla', 'dalla', 'match'],
      ['scandinavia', 'scandinavia', 'match'],
      ['ma', 'ma', 'match'],
      ['anche', 'anche', 'match'],
      ['dalla', 'dalla', 'match'],
      ['germania', 'germania', 'match'],
      ['e', 'e', 'match'],
      ['da', 'da', 'match'],
      ['altri', 'altri', 'match'],
      ['paesi', 'paesi', 'match'],
      ['europei', 'europei', 'match'],
      ['sebbene', 'sebbene', 'match'],
      ['i', 'i', 'match'],
      ['nani', 'nani', 'match'],
      ['siano', 'siano', 'match'],
      ['spesso', 'spesso', 'match'],
      ['di', 'di', 'match'],
      ['grande', 'grande', 'match'],
      ['aiuto', 'aiuto', 'match'],
      ['per', 'per', 'match'],
      ['gli', 'gli', 'match'],
      ['esseri', None, 'deletion'],
      ['umani', None, 'deletion'],
      ['e', None, 'deletion'],
      ['in', None, 'deletion'],
      ['questo', None, 'deletion'],
      ['caso', None, 'deletion'],
      ['sono', None, 'deletion'],
      ['chiamati', None, 'deletion'],
      ['spiriti', None, 'deletion'],
      ['buoni', None, 'deletion'],
      ['del', None, 'deletion'],
      ['focolare', None, 'deletion'],
      ['ci', None, 'deletion'],
      ['sono', None, 'deletion'],
      ['tra', None, 'deletion'],
      ['loro', None, 'deletion'],
      ['anche', None, 'deletion'],
      ['dei', None, 'deletion'],
      ['guerrieri', None, 'deletion']]}




```python

```
