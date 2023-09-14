import pandas as pd
import streamlit as st
import os
import requests
from st_clickable_images import clickable_images
from pytube import YouTube
from time import sleep

upload_endpoint = "https://api.assemblyai.com/v2/upload"
transcript_endpoint = "https://api.assemblyai.com/v2/transcript"

# https://www.assemblyai.com/
headers = {
    "authorization": "Enter your api key",
    "content-type": "application/json",
}


@st.cache_data
def save_audio(url):
    yt = YouTube(url=url)
    video = yt.streams.filter(only_audio=True).first()
    out_file = video.download(output_path="./audio files")
    base, ext = os.path.splitext(out_file)
    file_name = base + ".mp3"
    os.rename(out_file, file_name)
    print(yt.title + " has been successfully download")
    # print(file_name)
    return yt.title, file_name, yt.thumbnail_url


# """
# The first step is to upload this audio file to AssemblyAI with the function upload_to_AssemblyAI.
# Using the helper function read_file, the function reads the audio file in the given location (save_location) in chunks.
# This is used in the post request that is sent to the upload_endpoint of AssemblyAI together with the header for authentication.
# As a response, we get the URL to where the audio file is uploaded.
# """


@st.cache_data
def upload_to_AssemblyAI(save_location):
    CHUNK_SIZE = 5242880

    def read_file(filename):
        with open(filename, "rb") as _file:
            while True:
                print("chunk uploaded")
                data = _file.read(CHUNK_SIZE)
                if not data:
                    break
                yield data

    upload_response = requests.post(
        upload_endpoint,
        headers=headers,
        data=read_file(save_location)
    )
    print(upload_response.json())

    audio_url = upload_response.json()["upload_url"]
    print("Upload to", audio_url)


# """
# Here we'll use three AssemblyAI models:
# --The Summarization model—to return the summary of this audio file;
# --The Content Moderation model—to flag potentially sensitive and harmful content on topics such as alcohol, violence, gambling, and hate speech;
# --The Topic Detection model—to detect up to 700 topics (automotive, business, technology, education, standardized tests, inflation, off-road vehicles, and so on);

# The summarization model can give we different summaries:
# --A bullet list (bullets)
# --A longer bullet list (bullets_verbose)
# --A few words (gist)
# --A sentence (headline)
# --A paragraph (paragraph)

# The analysis will take a few seconds or minutes, depending on the length of the audio file. As a response to the transcription job, we'll get a job ID. Use it to create a polling endpoint to receive the analysis results.
# """


@st.cache_data
def start_analysis(audio_url):
    data = {
        "audio_url": audio_url,
        "iab_categories": True,
        "content_safety": True,
        "summarization": True,
        "summary_type": "bullets"
    }

    transcript_response = requests.post(
        transcript_endpoint,
        json=data,
        headers=headers
    )
    print(transcript_response)

    transcript_id = transcript_response.json()["id"]
    polling_endpoint = transcript_endpoint + "/" + transcript_id

    print("Transcribing to", polling_endpoint)
    return polling_endpoint


# """
# The last step is to collect the analysis results from AssemblyAI. The results are not generated instantaneously. Depending on the length of the audio file, the analysis might take a couple of seconds to a couple of minutes. To keep it simple and reusable, the process of receiving the analysis is wrapped in a function called get_analysis_results.

# In a while loop, every 10 seconds, a get request will be sent to AssemblyAI through the polling endpoint that includes the transaction job ID. In response to this get request, we'll get the job status as, "queued", “submitted”, “processing”, or “completed”.

# Once the status is "completed", the results are returned.
# """


@st.cache_data
def get_analysis_results(polling_endpoint):
    status = "submitted"

    while True:
        print(status)
        polling_response = requests.get(polling_endpoint, headers=headers)
        status = polling_response.json()["status"]
        # st.write(polling_response.json())

        if status == "submitted" or status == "processing" or status == "queued":
            print("Not ready yet")
            sleep(20)

        elif status == "completed":
            print("Creating transcript")
            return polling_response
            break

        else:
            print("Error")
            return False
            break


st.title("YouTube Content Analyzer")
st.markdown("With this app you can audit a Youtube channel to see if you'd like to sponsor them. All you have to do is to pass a list of links to the videos of this channel and you will get a list of thumbnails. Once you select a video by clicking its thumbnail, you can view:")
st.markdown("1. A summary of the video.", unsafe_allow_html=True)
st.markdown("2. The topics that are discussed in the video.",
            unsafe_allow_html=True)
st.markdown("3. Whether there are any sensitive topics discussed in the video.",
            unsafe_allow_html=True)
st.markdown("Make sure your links are in the format: https://www.youtube.com/watch?v=HfNnuQOHAaw and not https://youtu.be/HfNnuQOHAaw.", unsafe_allow_html=True)

default_bool = st.checkbox("Use a default file")

if default_bool:
    file = open("./links.txt")
else:
    file = st.file_uploader("Upload a file that includes the links (.txt)")

if file is not None:
    dataframe = pd.read_csv(file, header=None)
    dataframe.columns = ["urls"]
    urls_list = dataframe["urls"].tolist()

    titles = []
    locations = []
    thumbnails = []

    for video_url in urls_list:
        video_title, save_location, video_thumbnail = save_audio(video_url)
        titles.append(video_title)
        locations.append(save_location)
        thumbnails.append(video_thumbnail)

    selected_video = clickable_images(
        thumbnails,
        titles=titles,
        div_style={
            "height": "400px",
            "display": "flex",
            "justify-content": "center",
            "flex-wrap": "wrap",
            "overflow-y": "auto"
        },
        img_style={
            "margin": "5px",
            "height": "150px"
        }
    )

    st.markdown(
        f"Thumbnail #{selected_video} clicked" if selected_video > -1
        else "No image clicked"
    )

    if selected_video > -1:
        video_url = urls_list[selected_video]
        video_title = titles[selected_video]
        save_location = locations[selected_video]

        st.header(video_title)
        st.audio(save_location)

        audio_url = upload_to_AssemblyAI(save_location)

        polling_endpoint = start_analysis(audio_url)

        results = get_analysis_results(polling_endpoint)

        """
        We get three types of analysis on audio:

        --Summarization;
        --Sensitive content detection;
        --Topic detection;

        Extract the information with the “summary” keyword for the summarization results, “content_safety_labels” for content moderation and “iab_categories_result” for topic detection. Here is an example response:

        {
            "audio_duration": 1282,
            "confidence": 0.9414384528795772,
            "id": "oeo5u25f7-69e4-4f92-8dc9-f7d8ad6cdf38",
            "status": "completed",
            "text": "Ted talks are recorded live at the Ted Conference. This episode features...",
            "summary": "- Dan Gilbert is a psychologist and a happiness expert. His talk is recorded live at Ted conference. He explains why the human brain has nearly tripled in size in 2 million years. He also explains the difference between winning the lottery and becoming a paraplegic.\\n- In 1994, Pete Best said he's happier than he would have been with the Beatles. In the free choice paradigm, monet prints are ranked from the one they like the most to the one that they don't. People prefer the third one over the fourth one because it's a little better.\\n- People synthesize happiness when they change their affective. Hedonic aesthetic reactions to a poster. The ability to make up your mind and change your mind is the friend of natural happiness. But it's the enemy of synthetic happiness. The psychological immune system works best when we are stuck. This is the difference between dating and marriage. People don't know this about themselves and it can work to their disadvantage.\\n- In a photography course at Harvard, 66% of students choose not to take the course where they have the opportunity to change their mind. Adam Smith said that some things are better than others. Dan Gilbert recorded at Ted, 2004 in Monterey, California, 2004.",
            "content_safety_labels": {
                "status": "success",
                "results": [
                    {
                        "text": "Yes, that's it. Why does that happen? By calling off the Hunt, your brain can stop persevering on the ugly sister, giving the correct set of neurons a chance to be activated. Tip of the tongue, especially blocking on a person's name, is totally normal. 25 year olds can experience several tip of the tongues a week, but young people don't sweat them, in part because old age, memory loss, and Alzheimer's are nowhere on their radars.",
                        "labels": [
                            {
                                "label": "health_issues",
                                "confidence": 0.8225132822990417,
                                "severity": 0.15090347826480865
                            }
                        ],
                        "timestamp": {
                            "start": 358346,
                            "end": 389018
                        }
                    },
                    ...
                ],
                "summary": {
                    "health_issues": 0.8750781728032808
                    ...
                },
                "severity_score_summary": {
                    "health_issues": {
                        "low": 0.7210625030587972,
                        "medium": 0.2789374969412028,
                        "high": 0.0
                    }
                }
            },
            "iab_categories_result": {
                "status": "success",
                "results": [
                    {
                        "text": "Ted Talks are recorded live at Ted Conference...",
                        "labels": [
                            {
                                "relevance": 0.0005944414297118783,
                                "label": "Religion&Spirituality>Spirituality"
                            },
                            {
                                "relevance": 0.00039072768413461745,
                                "label": "Television>RealityTV"
                            },
                            {
                                "relevance": 0.00036419558455236256,
                                "label": "MusicAndAudio>TalkRadio>EducationalRadio"
                            }
                        ],
                        "timestamp": {
                            "start": 8630,
                            "end": 32990
                        }
                    },
                    ...
                ],
                "summary": {
                    "MedicalHealth>DiseasesAndConditions>BrainAndNervousSystemDisorders": 1.0,
                    "FamilyAndRelationships>Dating": 0.7614801526069641,
                    "Shopping>LotteriesAndScratchcards": 0.6330153346061707,
                    "Hobbies&Interests>ArtsAndCrafts>Photography": 0.6305723786354065,
                    "Style&Fashion>Beauty": 0.5269057750701904,
                    "Education>EducationalAssessment": 0.49798518419265747,
                    "BooksAndLiterature>ArtAndPhotographyBooks": 0.45763808488845825,
                    "FamilyAndRelationships>Bereavement": 0.45646440982818604,
                    "FineArt>FineArtPhotography": 0.3921416699886322,
                }
        }
        """

        summary = results.json()["summary"]
        topics = results.json()["iab_categories_result"]["summary"]
        sensitive_topics = results.json()["content_safety_labels"]["summary"]

        st.header("Summary of this video")
        st.write(summary)

        st.header("Sensitive content")
        if sensitive_topics != {}:
            st.subheader(
                "🚨 Mention of the following sensitive topics detected."
            )
            moderation_df = pd.DataFrame(sensitive_topics.items())
            moderation_df.columns = ["topic", "confidence"]
            st.dataframe(moderation_df, use_container_width=True)
        else:
            st.subheader("✅ All clear! No sensitive content detected.")

        st.header("Topics discussed")
        topics_df = pd.DataFrame(topics.items())
        topics_df.columns = ["topic", "confidence"]
        topics_df["topic"] = topics_df["topic"].str.split(">")
        expanded_topics = topics_df["topic"].apply(pd.Series).\
            add_prefix("topic_level_")
        topics_df = topics_df.join(expanded_topics).drop(columns="topic", axis=1).\
            sort_values(by=["confidence"], ascending=False).fillna("")

        st.dataframe(topics_df)
