from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import whisper
from pytube import YouTube
import os
import pandas as pd
from tqdm import tqdm
import traceback
import datetime
from datetime import datetime

time = datetime.now().strftime("%Y-%m-%d_%H%M%S")

#get transcritps for a video
def get_transcript(video_id, timeNow = time):
    #use youtube_transcript_api to get transcript
    transcripts = ""
    try:
      transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
      try:
        print ("getting transcript for ", video_id)
        transcript_en = transcript_list.find_transcript(['en']).fetch()
      except:
        for transcript in transcript_list:
          transcript_en = transcript.translate('en').fetch()
          for item in transcript_en:
            transcripts+= item["text"] + " "
          break
        return transcripts
      for word in transcript_en:
        if "text" in word:
          transcripts+= word["text"] + " "
        else:
          transcripts+= word + " "
    # if TranscriptsDisabled or NoTranscriptFound error, use alternative method
    except TranscriptsDisabled as e:
      print("Transcripts are disabled for this video, trying alternative method.")
      transcripts = transcribe_audio(video_id)
    except NoTranscriptFound as e:
      print("No transcript found for this video, trying alternative method.")
      transcripts = transcribe_audio(video_id)
    except Exception as e:
      print("type of exception", type(e))
      print(e)
      print(traceback.format_exc())
      print(e.args)
      print(e.with_traceback)
      transcripts = "Unexpectd Error occured"

    clean_transcripts(transcripts)
    save_transcripts_to_csv(transcripts, timeNow)

    return transcripts

def save_transcripts_to_csv(transcripts, timeNow):
    pd.DataFrame.from_dict(transcripts, orient='index', columns=['transcript']).reset_index().rename(columns={'index':'video_id'}).to_csv(f'transcripts_{timeNow}.csv')

def clean_transcripts(transcripts):
  try:
      # transcripts['transcript'] = transcripts['transcript'].str.replace(r'\n', ' ')
      transcripts = {k: v.replace(r'\n', ' ') for k, v in transcripts.items()}
      #remove commas
      transcripts = {k: v.replace(r',', ';') for k, v in transcripts.items()}
  except Exception as e:
      print(e)
      print('error in replacing new lines')
  return transcripts

#get transcripts from a list of videos
def get_transcripts(video_ids):
    transcripts = {}
    for video_id in video_ids:
        transcripts[video_id] = get_transcript(video_id)
    return transcripts

def transcribe_audio(video_id):
  transcription = ""

  video_url = "https://www.youtube.com/watch?v=" + video_id
  try:
    audio = get_audio(video_url)
    transcription = get_text(audio)
  except Exception as e:
    print(e)
    print(traceback.format_exc())
    # Live show, Not available, removed
    transcription = "Not Video"

  return transcription

def get_audio(url):
  yt = YouTube(url)
  video = yt.streams.filter(only_audio=True).first()
  out_file=video.download(output_path="videos\\.")
  base, ext = os.path.splitext(out_file)
  print('extension is ', ext)
  new_file = base+'.mp3'
  os.rename(out_file, new_file)
  return new_file

def get_text(file):
  model = whisper.load_model("base")
  result = model.transcribe(file, fp16=False, task ='translate')
  return result['text']