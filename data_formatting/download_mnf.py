import os
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from config import j

home_folder = os.environ['HOME']

# Define the scopes
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Obtain your Google credentials
def get_credentials():
    flow = service_account.Credentials.from_service_account_file(os.path.join(home_folder, 'credentials.json'), scopes=SCOPES)
    return flow

# Build the downloader
creds = get_credentials()
drive_downloader = build('drive', 'v3', credentials=creds)

# Replace 'FOLDER_ID' with your actual Google Drive folder ID
folder_id = '1u-6mWw8qOLrGR35zX6lGGn30Z1t8V6cf'


# Get the list of files in the folder
query = f"'{folder_id}' in parents"
results = drive_downloader.files().list(q=query, pageSize=1000).execute()
items = results.get('files', [])

# Download all files
def download_files(item):
    file_id = item['id']
    file_name = item['name']
    request = drive_downloader.files().get_media(fileId=file_id)

    # Save the file to the current working directory
    fh = io.FileIO(j(f'MNF/{file_name}'), 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print(f"Download {file_name[7:]} {int(status.progress() * 100)}%")


for item in items:
    download_files(item)