from barcode_script.utils.utils import download_video
from barcode_script.utils.barcode import vid2barcode
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import concurrent
from tqdm import tqdm
import glob
import os


def collect_barcode(orig_video_ids, filename):

  video_ids = list(orig_video_ids)
  path = 'barcode_script/scripts/videos/' + filename + '/*.3gpp'
  if len(glob.glob(path)) != 0:
      video_files = []
      for file_path in glob.glob(path):
          video_files.append(os.path.basename(file_path))

      videos = []
      for video in video_files:
          videos.append(video.split('.')[0])

      existing = 0
      existed_videos = []
      for video_id in video_ids:
          if video_id in videos:
              existed_videos.append(video_id)
              existing +=1

      # Remove
      for video_id in existed_videos:
          video_ids.remove(video_id)

      print('\n Skipped download for '+ str(existing) + ' videos')
  else:
      print('No existing video is found. Downloading all videos.')

  # Multi-threading for downloading videos
  with ThreadPoolExecutor(max_workers=8) as executor:
      futures = [executor.submit(download_video, video_id, filename) for video_id in video_ids]
      for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Downloading videos'):
          # Optionally handle errors here
          try:
              future.result()
          except Exception as e:
              print(f"Error downloading video: {e}")

  #download_video('dKrVegVI0Us')

  # get filenames in videos folder
  # run videos to barcodes

  barcode_path = 'barcode_script/scripts/barcode/' + filename + '/*.png'
  if len(glob.glob(barcode_path)) != 0:
      barcode_files = []
      for file_path in glob.glob(barcode_path):
          barcode_files.append(os.path.basename(file_path))

      barcodes = []
      for barcode in barcode_files:
          barcodes.append(barcode.split('.')[0])

      existing = 0
      existed_barcodes = []
      for video_id in orig_video_ids:
          if video_id in barcodes:
              existed_barcodes.append(video_id)
              existing +=1

      # Remove
      barcode_videos_ids = list(orig_video_ids)
      for video_id in existed_barcodes:
          barcode_videos_ids.remove(video_id)

      video_files = []
      for video_id in barcode_videos_ids:
          if video_id not in video_ids:
              video_files.append(video_id+'.3gpp')

      print('\n Skipped generating for ' + str(existing) + ' barcodes')
  else:
      print('No existing barcode is found. Generating all barcodes.')
      video_files = []
      for file_path in glob.glob(path):
          video_files.append(os.path.basename(file_path))

  # Multi-processing for generating barcodes
  with ProcessPoolExecutor(max_workers=10) as executor:
      futures = [executor.submit(vid2barcode, file, filename) for file in video_files]
      for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Generating barcodes'):
          # Optionally handle errors here
          try:
              future.result()
          except Exception as e:
              print(f"Error generating barcode: {e}")

