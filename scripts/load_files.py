from imports import import_cv2, import_moviepy
import pandas as pd

cv2 = import_cv2()
moviepy, VideoFileClip = import_moviepy()

root_data_dir = '/net/projects/ycleong/users/dsi_sl_2025/algonauts_tutorial/data'
print(f"root_data_dir is set to: {root_data_dir}")

def load_mkv_file(movie_path):
    """
    Load video and audio data from the given .mkv movie file, and additionally
    prints related information.

    Parameters
    ----------
    movie_path : str
        Path to the .mkv movie file.

    """

    # Read the .mkv file
    cap = cv2.VideoCapture(movie_path)

    if not cap.isOpened():
        print("Error: Could not open movie.")
        return

    # Get video information
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = video_total_frames / video_fps
    video_duration_minutes = video_duration / 60

    # Release the video object
    cap.release()

    # Audio information
    clip = VideoFileClip(movie_path)
    audio = clip.audio
    audio_duration = audio.duration
    audio_fps = audio.fps

    print(f"Video FPS: {video_fps}")
    print(f"Video size: {video_width}x{video_height}")
    print(f"Total frames: {video_total_frames}")
    print(f"Video duration (s): {video_duration:.2f} ({video_duration_minutes:.2f} min)")
    print(f"Audio duration (s): {audio_duration:.2f}")
    print(f"Audio FPS: {audio_fps}")


# Load the .mkv file
movie_path = root_data_dir + "/stimuli/movies/friends/s1/friends_s01e01a.mkv"
load_mkv_file(movie_path)


###############################


def load_tsv_file(transcript_path):
    """
    Load and visualize language transcript data from the given .TSV file.

    Parameters
    ----------
    transcript_path : str
        Path to the .tsv transcript file.

    """

    # Load the .tsv into a pandas DataFrame
    transcript_df = pd.read_csv(transcript_path, sep='\t')

    # Select the first 20 rows (chunks)
    sample_transcript_data = transcript_df.iloc[:20]


# Load the .tsv file
transcript_path = root_data_dir + "/stimuli/transcripts/friends/s1/friends_s01e01a.tsv"
load_tsv_file(transcript_path)


###############################


def load_transcript(transcript_path):
    """
    Loads a transcript file and returns it as a DataFrame.

    Parameters
    ----------
    transcript_path : str
        Path to the .tsv transcript file.

    """
    df = pd.read_csv(transcript_path, sep='\t')
    print(f"Loaded transcript with {len(df)} rows")
    return df


def get_movie_info(movie_path):
    """
    Extracts the frame rate (FPS) and total duration of a movie.

    Parameters
    ----------
    movie_path : str
        Path to the .mkv movie file.

    """

    cap = cv2.VideoCapture(movie_path)
    fps, frame_count = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    return fps, frame_count / fps


def split_movie_into_chunks(movie_path, chunk_duration=1.49):
    """
    Divides a video into fixed-duration chunks.

    Parameters
    ----------
    movie_path : str
        Path to the .mkv movie file.
    chunk_duration : float, optional
        Duration of each chunk in seconds (default is 1.49).

    """

    _, video_duration = get_movie_info(movie_path)
    chunks = []
    start_time = 60.0

    # Create chunks for the specified time
    while start_time < video_duration:
        end_time = min(start_time + chunk_duration, video_duration)
        chunks.append((start_time, end_time))
        start_time += chunk_duration
    return chunks

def extract_movie_segment_with_sound(movie_path, start_time, end_time,
    output_path='output_segment.mp4'):
    """
    Extracts a specific segment of a video with sound and saves it.

    Parameters
    ----------
    movie_path : str
        Path to the .mkv movie file.
    start_time : float
        Start time of the segment in seconds.
    end_time : float
        End time of the segment in seconds.
    output_path : str, optional
        Path to save the output segment (default is 'output_segment.mp4').

    """

    print(f"Extracting segment from {start_time:.2f}s to {end_time:.2f}s into {output_path}")
    # Create movie segment
    movie_segment = VideoFileClip(movie_path).subclip(start_time, end_time)

    # Write video file
    movie_segment.write_videofile(output_path, codec="libx264",
        audio_codec="aac", verbose=False, logger=None)
    print("Extraction complete")
    return output_path




# Base directory for fMRI data
fmri_dir = root_data_dir + "/fmri"
print(f"fMRI directory set to: {fmri_dir}")
