
import streamlit as st
import numpy as np
import pandas as pd
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
import tqdm

# Device configuration (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained model
model_path = 'C://Users//melon//OneDrive//Bureau//appfr//camembert_model2.pth'
tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
model = CamembertForSequenceClassification.from_pretrained('camembert-base', num_labels=6)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# YouTube API Key
API_KEY = 'AIzaSyC-Q4Gm7P1f1enRsg0nUJ8FwWquGH0KkBc'

# Initialize the YouTube client
youtube = build('youtube', 'v3', developerKey=API_KEY)

def bert_feature(data, **kwargs):
    input_ids = [tokenizer.encode(text, add_special_tokens=True, **kwargs) for text in data]
    features = []
    with torch.no_grad():
        for input_id in tqdm.tqdm(input_ids):
            if len(input_id) == 0:
                continue
            input_tensor = torch.tensor(input_id).unsqueeze(0).to(device)
            outputs = model(input_tensor)
            feature = outputs.logits.cpu().numpy()
            features.append(feature)
    if len(features) == 0:
        return np.array([])
    feature_data = np.concatenate(features, axis=0)
    return feature_data

def get_video_details(youtube, **kwargs):
    return youtube.videos().list(
        part="snippet,contentDetails,statistics",
        **kwargs
    ).execute()

def get_video_infos(video_response):
    items = video_response.get("items")[0]
    snippet = items["snippet"]
    title = snippet["title"]
    description = snippet["description"]
    thumbnails_url = snippet['thumbnails']['default']['url']
    return title, description, thumbnails_url

def search(youtube, **kwargs):
    return youtube.search().list(
        part="snippet",
        type="video",
        **kwargs
    ).execute()

def retrieve_video_list(keyword):
    response = search(youtube, q=keyword, maxResults=50, relevanceLanguage="fr", videoCaption="closedCaption")
    items = response.get("items", [])
    if not items:
        st.warning("No videos found for this keyword.")
        return pd.DataFrame()
    df_video = pd.DataFrame(columns=['video url', 'title', 'description', 'thumbnails url', 'caption'])
    for item in tqdm.tqdm(items):
        video_id = item["id"]["videoId"]
        video_response = get_video_details(youtube, id=video_id)
        title, description, thumbnails_url = get_video_infos(video_response)
        try:
            srt = YouTubeTranscriptApi.get_transcript(video_id, languages=['fr'])
        except:
            try:
                srt = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            except:
                continue
        caption = '. '.join([transcript['text'] for transcript in srt])
        caption = caption.replace('\n', ' ')
        video = {'video url': 'https://www.youtube.com/watch?v=' + video_id, 'title': title, 'description': description, 'thumbnails url': thumbnails_url, 'caption': caption}
        df_video = pd.concat([df_video, pd.DataFrame([video])], ignore_index=True)
    return df_video

def predictor(keyword, level):
    df_video = retrieve_video_list(keyword)
    if df_video.empty:
        return pd.DataFrame()
    test_features = bert_feature(df_video['caption'], max_length=512)
    if test_features.size == 0:
        return pd.DataFrame()
    pred_difficulty = test_features.argmax(axis=1)
    df_video['difficulty'] = pd.Series(pred_difficulty).map({0:'A1', 1:'A2', 2:'B1', 3:'B2', 4:'C1', 5:'C2'})
    df_result = df_video[df_video['difficulty'] == level].reset_index()
    return df_result

# Streamlit User Interface
st.title('French YouTube Video Finder')
st.subheader("This application helps you find French YouTube videos matching a specific language difficulty level of your choice!")

keyword_input = st.text_input('Enter a keyword:')
level_dropdown = st.selectbox('Choose a difficulty level:', ['A1', 'A2', 'B1', 'B2', 'C1', 'C2'])
show_videos_button = st.button('Show Recommended Videos')

def show_videos():
    st.empty()
    keyword = keyword_input
    level = level_dropdown
    with st.spinner('Loading...'):
        df_result = predictor(keyword, level)
    if df_result.empty:
        st.error('No matches found for your input! Please change your keyword or level! (Note: Most videos are above the C1 level)')
        return
    if len(df_result) == 1:
        st.success('There is one recommended video for you!')
    else:
        st.success(f'There are {len(df_result)} recommended videos for you!')
    st.write('\n')
    for i in range(len(df_result)):
        st.write(f"{i + 1}. Title: {df_result['title'][i]}")
        st.write(f"URL: {df_result['video url'][i]}")
        st.image(df_result['thumbnails url'][i])
        st.write('\n')

if show_videos_button:
    show_videos()