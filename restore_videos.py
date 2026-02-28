#!/usr/bin/env python3
# restore_videos.py
#
# Utility to restore videos from processed/skipped folders back to the main folder

import shutil
from pathlib import Path
import argparse


def restore_videos(video_folder: str, folder_type: str = "both"):
    """
    Restore videos from processed/skipped folders back to main folder.
    
    Args:
        video_folder: Main video folder
        folder_type: Which folder to restore from ("processed", "skipped", or "both")
    """
    video_folder = Path(video_folder)
    
    if not video_folder.exists():
        print(f"❌ Video folder not found: {video_folder}")
        return
    
    restored_count = 0
    
    # Restore from processed folder
    if folder_type in ["processed", "both"]:
        processed_folder = video_folder / "processed"
        if processed_folder.exists():
            videos = list(processed_folder.glob("*.mp4")) + \
                    list(processed_folder.glob("*.MP4")) + \
                    list(processed_folder.glob("*.mov")) + \
                    list(processed_folder.glob("*.MOV")) + \
                    list(processed_folder.glob("*.avi")) + \
                    list(processed_folder.glob("*.AVI"))
            
            print(f"\n📁 Found {len(videos)} videos in processed folder")
            
            if videos:
                confirm = input(f"Restore {len(videos)} videos from processed? (y/N): ").strip().lower()
                if confirm == 'y':
                    for video in videos:
                        try:
                            destination = video_folder / video.name
                            
                            # Handle duplicates
                            if destination.exists():
                                base = destination.stem
                                ext = destination.suffix
                                counter = 1
                                while destination.exists():
                                    destination = video_folder / f"{base}_restored_{counter}{ext}"
                                    counter += 1
                            
                            shutil.move(str(video), str(destination))
                            print(f"  ✅ {video.name} → {destination.name}")
                            restored_count += 1
                        except Exception as e:
                            print(f"  ❌ Failed to restore {video.name}: {e}")
    
    # Restore from skipped folder
    if folder_type in ["skipped", "both"]:
        skipped_folder = video_folder / "skipped"
        if skipped_folder.exists():
            videos = list(skipped_folder.glob("*.mp4")) + \
                    list(skipped_folder.glob("*.MP4")) + \
                    list(skipped_folder.glob("*.mov")) + \
                    list(skipped_folder.glob("*.MOV")) + \
                    list(skipped_folder.glob("*.avi")) + \
                    list(skipped_folder.glob("*.AVI"))
            
            print(f"\n📁 Found {len(videos)} videos in skipped folder")
            
            if videos:
                confirm = input(f"Restore {len(videos)} videos from skipped? (y/N): ").strip().lower()
                if confirm == 'y':
                    for video in videos:
                        try:
                            destination = video_folder / video.name
                            
                            # Handle duplicates
                            if destination.exists():
                                base = destination.stem
                                ext = destination.suffix
                                counter = 1
                                while destination.exists():
                                    destination = video_folder / f"{base}_restored_{counter}{ext}"
                                    counter += 1
                            
                            shutil.move(str(video), str(destination))
                            print(f"  ✅ {video.name} → {destination.name}")
                            restored_count += 1
                        except Exception as e:
                            print(f"  ❌ Failed to restore {video.name}: {e}")
    
    print(f"\n✅ Restored {restored_count} videos total")


def list_archived_videos(video_folder: str):
    """List all archived videos without restoring."""
    video_folder = Path(video_folder)
    
    print("\n" + "="*70)
    print("ARCHIVED VIDEOS")
    print("="*70)
    
    # List processed videos
    processed_folder = video_folder / "processed"
    if processed_folder.exists():
        videos = list(processed_folder.glob("*.mp4")) + \
                list(processed_folder.glob("*.MP4")) + \
                list(processed_folder.glob("*.mov")) + \
                list(processed_folder.glob("*.MOV")) + \
                list(processed_folder.glob("*.avi")) + \
                list(processed_folder.glob("*.AVI"))
        
        print(f"\n📦 PROCESSED ({len(videos)} videos):")
        for video in sorted(videos):
            print(f"  - {video.name}")
    
    # List skipped videos
    skipped_folder = video_folder / "skipped"
    if skipped_folder.exists():
        videos = list(skipped_folder.glob("*.mp4")) + \
                list(skipped_folder.glob("*.MP4")) + \
                list(skipped_folder.glob("*.mov")) + \
                list(skipped_folder.glob("*.MOV")) + \
                list(skipped_folder.glob("*.avi")) + \
                list(skipped_folder.glob("*.AVI"))
        
        print(f"\n⏭️  SKIPPED ({len(videos)} videos):")
        for video in sorted(videos):
            print(f"  - {video.name}")
    
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restore archived videos")
    parser.add_argument("--videos", "-v", required=True,
                       help="Main video folder")
    parser.add_argument("--from", dest="folder_type", 
                       choices=["processed", "skipped", "both"],
                       default="both",
                       help="Which folder to restore from")
    parser.add_argument("--list", "-l", action="store_true",
                       help="Just list archived videos without restoring")
    
    args = parser.parse_args()
    
    if args.list:
        list_archived_videos(args.videos)
    else:
        restore_videos(args.videos, args.folder_type)