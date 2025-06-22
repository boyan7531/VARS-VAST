#!/usr/bin/env python3
"""
Fix Corrupted Videos - Find and optionally remove corrupted videos from MVFouls dataset
"""

import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_corrupted_videos(data_dir: Path, splits: list = None):
    """Find all corrupted (small size) videos in the dataset."""
    if splits is None:
        splits = ['train', 'test', 'valid', 'challenge']
    
    corrupted_videos = []
    total_videos = 0
    
    for split in splits:
        split_dir = data_dir / f"{split}_720p"
        
        if not split_dir.exists():
            logger.info(f"Split directory {split_dir} does not exist, skipping")
            continue
        
        logger.info(f"Checking {split} split...")
        
        # Find all video files
        for action_dir in split_dir.iterdir():
            if action_dir.is_dir() and action_dir.name.startswith('action_'):
                for video_file in action_dir.glob("*.mp4"):
                    total_videos += 1
                    
                    try:
                        file_size = video_file.stat().st_size
                        # Consider videos under 1KB as corrupted (normal videos are MB in size)
                        if file_size < 1024:
                            corrupted_videos.append(video_file)
                            logger.warning(f"Found corrupted video ({file_size} bytes): {video_file}")
                    except Exception as e:
                        logger.error(f"Error checking {video_file}: {e}")
                        corrupted_videos.append(video_file)
    
    return corrupted_videos, total_videos

def remove_corrupted_videos(corrupted_videos: list, dry_run: bool = True):
    """Remove corrupted videos from the filesystem."""
    if not corrupted_videos:
        logger.info("No corrupted videos to remove")
        return
    
    if dry_run:
        logger.info(f"DRY RUN: Would remove {len(corrupted_videos)} corrupted videos:")
        for video in corrupted_videos:
            logger.info(f"  Would remove: {video}")
    else:
        logger.info(f"Removing {len(corrupted_videos)} corrupted videos...")
        removed_count = 0
        
        for video in corrupted_videos:
            try:
                video.unlink()
                logger.info(f"Removed: {video}")
                removed_count += 1
            except Exception as e:
                logger.error(f"Failed to remove {video}: {e}")
        
        logger.info(f"Successfully removed {removed_count}/{len(corrupted_videos)} corrupted videos")

def main():
    parser = argparse.ArgumentParser(description='Find and fix corrupted videos in MVFouls dataset')
    parser.add_argument('--data-dir', type=str, default='mvfouls',
                       help='Path to mvfouls dataset directory')
    parser.add_argument('--splits', nargs='+', default=['train', 'test', 'valid'],
                       help='Dataset splits to check')
    parser.add_argument('--remove', action='store_true',
                       help='Actually remove corrupted videos (default: dry run)')
    parser.add_argument('--list-only', action='store_true',
                       help='Only list corrupted videos, do not remove')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        logger.error(f"Data directory {data_dir} does not exist")
        return
    
    logger.info(f"Scanning dataset for corrupted videos...")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Splits: {args.splits}")
    
    # Find corrupted videos
    corrupted_videos, total_videos = find_corrupted_videos(data_dir, args.splits)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 CORRUPTED VIDEO SCAN RESULTS")
    print(f"{'='*60}")
    print(f"Total videos scanned: {total_videos}")
    print(f"Corrupted videos found: {len(corrupted_videos)}")
    print(f"Corruption rate: {len(corrupted_videos)/total_videos*100:.2f}%")
    
    if corrupted_videos:
        print(f"\n❌ CORRUPTED VIDEOS:")
        for video in corrupted_videos:
            print(f"  - {video}")
        
        if not args.list_only:
            # Remove corrupted videos
            dry_run = not args.remove
            if dry_run:
                print(f"\n🔍 DRY RUN MODE (use --remove to actually delete)")
            
            remove_corrupted_videos(corrupted_videos, dry_run=dry_run)
            
            if dry_run:
                print(f"\n💡 To actually remove corrupted videos, run:")
                print(f"   python fix_corrupted_videos.py --data-dir {args.data_dir} --remove")
    else:
        print(f"\n✅ No corrupted videos found! Dataset is clean.")

if __name__ == "__main__":
    main() 