import json
import os
import urllib
import urllib.request
from urllib.request import urlopen
import re
import unicodedata
import requests
from collections import Counter
import argparse

from bs4 import BeautifulSoup
import pandas as pd

from youtube_transcript_api import YouTubeTranscriptApi
import youtube_dl

# # Set Environmental Variables

parser = argparse.ArgumentParser()
parser.add_argument("--error_filename_1", type=str, default="extract_errors_method1.txt", help="Filename for .txt to catch errors from extraction method 1")
parser.add_argument("--error_filename_2", type=str, default="extract_errors_method2.txt", help="Filename for .txt to catch errors from extraction method 2")
parser.add_argument("--error_filename_3", type=str, default="extract_errors_method3.txt", help="Filename for .txt to catch errors from extraction method 3")
parser.add_argument("--source_dir", type=str, default="../", help="Directory where source JSON files without text are located")
parser.add_argument("--target_dir", type=str, default="../text_extracted_datasets", help="Directory where finalized JSON files with text will be located")
args = parser.parse_args()

NUM_OF_ERRORS = 0

# # Create Target Directory if not already created
if not os.path.isdir(args.target_dir):
    os.mkdir(args.target_dir)


# ## Methods 
# ### 3 Methods to Pull Captions from Youtube Video via Python Libraries

def caption_extraction(video_id):
    """
    Extract YouTube Captions from Any Transcript
    Defaults to English & Always Picks Manually Created Transcripts over Automatically Created Ones
    """
    # Set variables to collect video caption
    full_text = ''
    global NUM_OF_ERRORS
    
    # Open text file to log errors
    f = open(args.error_filename_1, 'a')
    
    try:
        
        srt = YouTubeTranscriptApi.get_transcript(video_id, cookies='youtube.com_cookies.txt') 
        # Downloaded ...youtube.com_cookies file from age-restricted YouTube Video with UID 'vOCf8MhnQBs' via Mozilla Firefox Extension 'Get youtube.com_cookies' 
        # as an attempt to download transcripts from age-restricted videos - sometimes does not work
        full_text += ' '.join(map(str, [dct['text'] for dct in srt]))
        return full_text
    
    # Specific except statement for NoTranscriptFound errors
    # When no manually-created or automatically-created english transcript, searches available transcripts and tries to translate them to English
    except:
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            for transcript in transcript_list:
                # Check if English is an option for translation in current transcript, else go to the next one
                if not {'language': 'English', 'language_code': 'en'} in transcript.translation_languages:
                    continue
                srt = transcript.translate('en').fetch()
                full_text += ' '.join(map(str, [dct['text'] for dct in srt]))
                return full_text
            
        # Log any Errors
        except Exception as e:
            NUM_OF_ERRORS += 1
            print('\nError ' + str(NUM_OF_ERRORS) + ':', file=f)

            print('YouTube UID: ' + str(video_id), file=f) # Print links to any Youtube video of which captions were not extracted
            print(e, file=f)
            pass
            
    f.close()

def caption_extraction_v2(video_id):
    """
    Extract YouTube Captions via YouTube DL
    Specific to English - extracts from any video (even age-restricted) with manual, automatic, translatable-to-English captions
    Does not extract video that is private or unavailable
    """
    # Set variables to collect video caption
    full_text = ''
    global NUM_OF_ERRORS
    
    # Open text file to log errors
    f = open(args.error_filename_2, 'a')
    
    try:
        ydl = youtube_dl.YoutubeDL({'writesubtitles': True, 'allsubtitles': True, 'writeautomaticsub': True, 'cookiefile': 'youtube.com_cookies.txt'})
        res = ydl.extract_info(video_id, download=False)
        if res['requested_subtitles'] and res['requested_subtitles']['en']:
            print('Grabbing vtt file from ' + res['requested_subtitles']['en']['url'])
            response = requests.get(res['requested_subtitles']['en']['url'], stream=True)
            new = re.sub(r'\d{2}\W\d{2}\W\d{2}\W\d{3}\s\W{3}\s\d{2}\W\d{2}\W\d{2}\W\d{3}','',response.text)
            new = re.sub("<.*?>", "", new)
            new = re.sub("align:start position:0%", "", new)
            if len(res['subtitles']) > 0:
                print('manual captions')
                return new.split("Language: en",1)[1] 
            else:
                print('automatic_captions')
                return new.split("Language: en",1)[1] 
        else:
            print('Youtube Video does not have any english captions')
            
    # Log any Errors
    except Exception as e:
        NUM_OF_ERRORS += 1
        print('\nError ' + str(NUM_OF_ERRORS) + ':', file=f)

        print('YouTube UID: ' + str(video_id), file=f) # Print links to any Youtube video of which captions were not extracted
        print(e, file=f)
        pass 
        
    f.close()

def caption_extraction_v3(video_id):
    """
    Extract YouTube Captions from Youtube-Transcript-Specific XML File Accessible via Web Browser URL
    Specific to English
    """
    # Set variables to collect video caption
    full_text = ''
    global NUM_OF_ERRORS
    
    # Open text file to log errors
    f = open(args.error_filename_3, 'a')
    
    try:
        html = urlopen("https://www.youtube.com/api/timedtext?&v=" + str(video_id) + "&lang=en")
        soup = BeautifulSoup(html,'html.parser')
        full_text += ' '.join(map(str, [captions.text.replace("&#39;", "'") for captions in soup.find_all('text')]))
        return full_text
    
    # Log any Errors
    except Exception as e:
        NUM_OF_ERRORS += 1
        print('\nError ' + str(NUM_OF_ERRORS) + ':', file=f)

        print('YouTube UID: ' + str(video_id), file=f) # Print links to any Youtube video of which captions were not extracted
        print(e, file=f)
        pass 
        
    f.close()


# # Pre-Process Data

# ## Download Full Dataset

datasets = {}
# Import JSON files from source directory
json_filenames = [pos_json for pos_json in os.listdir(args.source_dir) if pos_json.endswith('.json')]
for json_filename in json_filenames:
    datasets[json_filename] = pd.read_json(args.source_dir + json_filename)
    # Reset index
    datasets[json_filename].reset_index(drop=True, inplace=True)

# ## Extract Captions for YouTube Video with First Method
# 1. Create .txt file to log any errors
# 2. Add Captions from video to new Column labeled "video_sub_title"

# Create new text file to log errors if not already made
if not os.path.isfile(args.error_filename_1):
    open(args.error_filename_1, 'w').close()

# Run through every JSON File to extract captions for each file
for json_filename in json_filenames:
    
    # Log length of Dataset
    f = open(args.error_filename_1, 'a')
    print('\n' + json_filename[:-5].upper() + ' Dataset Length without Captions: ' + str(len(datasets[json_filename].index)), file=f)
    f.close()
    
    # Reset number of errors to log in text file
    NUM_OF_ERRORS = 0
    
    datasets[json_filename]['video_sub_title'] = datasets[json_filename]['video_id'].apply(caption_extraction)
    print('Number of Errors in Dataset: ' + str(NUM_OF_ERRORS))

# Check which videos were still not scraped
for json_filename in json_filenames:
    print('Number of Unextracted Vids in ' + json_filename[:-5].upper() + ' Dataset: ', str(len(datasets[json_filename][(datasets[json_filename]['video_sub_title'] == '') | (datasets[json_filename]["video_sub_title"].str.isspace()) | datasets[json_filename]['video_sub_title'].isna()])))


# ## Use Second Extraction Method to Fill in Blanks 

# Create new text file to log errors if not already made
if not os.path.isfile(args.error_filename_2):
    open(args.error_filename_2, 'w').close()

# Run through every JSON File to extract captions for each file
for json_filename in json_filenames:
    
    # Log length of Dataset
    f = open(args.error_filename_2, 'a')
    print('\n' + json_filename[:-5].upper() + ' Dataset Length without Captions: ' + str(len(datasets[json_filename].loc[(datasets[json_filename]['video_sub_title'] == '') | (datasets[json_filename]["video_sub_title"].str.isspace()) | (datasets[json_filename]['video_sub_title'].isna())].index)), file=f)
    f.close()
    
    # Reset number of errors to log in text file
    NUM_OF_ERRORS = 0
    
    
    datasets[json_filename].loc[(datasets[json_filename]['video_sub_title'] == '') | (datasets[json_filename]["video_sub_title"].str.isspace()) | (datasets[json_filename]['video_sub_title'].isna()), 'video_sub_title'] = datasets[json_filename].loc[(datasets[json_filename]['video_sub_title'] == '') | (datasets[json_filename]["video_sub_title"].str.isspace()) | (datasets[json_filename]['video_sub_title'].isna()), 'video_id'].apply(caption_extraction_v2)
    print('Number of Errors in Dataset: ' + str(NUM_OF_ERRORS))

# Check which videos were still not scraped
for json_filename in json_filenames:
    print('Number of Unextracted Vids in ' + json_filename[:-5].upper() + ' Dataset: ', str(len(datasets[json_filename][(datasets[json_filename]['video_sub_title'] == '') | (datasets[json_filename]["video_sub_title"].str.isspace()) | datasets[json_filename]['video_sub_title'].isna()])))

# ## Use Third Extraction Method to Fill in Blanks 

# Create new text file to log errors if not already made
if not os.path.isfile(args.error_filename_3):
    open(args.error_filename_3, 'w').close()

# Run through every JSON File to extract captions for each file
for json_filename in json_filenames:
    
    # Log length of Dataset
    f = open(args.error_filename_3, 'a')
    print('\n' + json_filename[:-5].upper() + ' Dataset Length without Captions: ' + str(len(datasets[json_filename].loc[(datasets[json_filename]['video_sub_title'] == '') | (datasets[json_filename]["video_sub_title"].str.isspace()) | (datasets[json_filename]['video_sub_title'].isna())].index)), file=f)
    f.close()
    
    # Reset number of errors to log in text file
    NUM_OF_ERRORS = 0
    
    
    datasets[json_filename].loc[(datasets[json_filename]['video_sub_title'] == '') | (datasets[json_filename]["video_sub_title"].str.isspace()) | (datasets[json_filename]['video_sub_title'].isna()), 'video_sub_title'] = datasets[json_filename].loc[(datasets[json_filename]['video_sub_title'] == '') | (datasets[json_filename]["video_sub_title"].str.isspace()) | (datasets[json_filename]['video_sub_title'].isna()), 'video_id'].apply(caption_extraction_v3)
    print('Number of Errors in Dataset: ' + str(NUM_OF_ERRORS))

# Check which videos were still not scraped
for json_filename in json_filenames:
    print('Number of Unextracted Vids in ' + json_filename[:-5].upper() + ' Dataset: ', str(len(datasets[json_filename][(datasets[json_filename]['video_sub_title'] == '') | (datasets[json_filename]["video_sub_title"].str.isspace()) | datasets[json_filename]['video_sub_title'].isna()])))


# ## Perform the following tasks for all datasets:
# 1. Recreate Dataframe to have only 5 columns:
# 2. Clean Up Data
# 3. Send to JSON Files

for json_filename in json_filenames:
    
    # Create new dataframe to set to only 5 columns & clean up '\n' in video_sub_title
    processed_df = pd.DataFrame({
        'video_id': datasets[json_filename]['video_id'],
        'video_link': datasets[json_filename]['video_link'],
        'video_title': datasets[json_filename]['video_title'],
        'video_sub_title': datasets[json_filename]["video_sub_title"].replace(r'\n', ' ', regex=True),
        'label': datasets[json_filename]['label'],
    })
        
    # Remove trailing and leading white spaces
    processed_df['video_sub_title'] = processed_df['video_sub_title'].str.strip()

    # Reset index
    processed_df.reset_index(drop=True, inplace = True)
    
    # Change any non-UTF-8 symbols to UTF-8
    processed_df['video_sub_title'] = processed_df['video_sub_title'].apply(lambda val: unicodedata.normalize('NFKD', val))

    # Fill any empty cells with empty string
    processed_df = processed_df.fillna('')
    
    # Export File to JSON to Check Design
    processed_df.to_json(args.target_dir + '/' + json_filename, orient="records", indent=4)
