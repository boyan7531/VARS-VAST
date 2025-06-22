#!/usr/bin/env python3
"""
Dataset Validation Script for MVFouls Dataset

This script validates all video files in the MVFouls dataset by testing
different video loading methods and reporting any issues.

Usage:
    python validate_dataset.py --data-dir mvfouls --splits train test valid
    python validate_dataset.py --data-dir mvfouls --fix-corrupted
"""

import argparse
import logging
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
from collections import defaultdict

# Video loading libraries
import cv2
import numpy as np

try:
    import decord
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

try:
    import av
    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VideoValidator:
    """Validates video files using multiple loading methods."""
    
    def __init__(self, num_frames: int = 32, center_frame: int = 75):
        self.num_frames = num_frames
        self.center_frame = center_frame
        self.results = {
            'total_videos': 0,
            'successful_videos': 0,
            'failed_videos': 0,
            'corrupted_videos': [],
            'method_success': {
                'decord': 0,
                'pyav': 0,
                'opencv': 0
            },
            'method_failures': {
                'decord': [],
                'pyav': [],
                'opencv': []
            }
        }
    
    def test_decord(self, video_path: Path) -> Tuple[bool, str, Optional[np.ndarray]]:
        """Test video loading with decord."""
        if not DECORD_AVAILABLE:
            return False, "decord not available", None
        
        try:
            vr = decord.VideoReader(str(video_path))
            total_frames = len(vr)
            
            if total_frames == 0:
                return False, "No frames in video", None
            
            # Calculate frame indices
            start_frame = max(0, self.center_frame - self.num_frames // 2)
            end_frame = min(total_frames, start_frame + self.num_frames)
            frame_indices = list(range(start_frame, end_frame))
            
            if not frame_indices:
                return False, "No valid frame indices", None
            
            # Try to read frames
            frames = vr.get_batch(frame_indices).asnumpy()
            
            if frames.shape[0] == 0:
                return False, "Could not read any frames", None
            
            return True, f"Success: {frames.shape[0]} frames, resolution {frames.shape[1]}x{frames.shape[2]}", frames
            
        except Exception as e:
            return False, f"decord error: {str(e)}", None
    
    def test_pyav(self, video_path: Path) -> Tuple[bool, str, Optional[np.ndarray]]:
        """Test video loading with PyAV."""
        if not PYAV_AVAILABLE:
            return False, "PyAV not available", None
        
        try:
            container = av.open(str(video_path))
            video_stream = container.streams.video[0]
            
            frames = []
            frame_count = 0
            target_frames = set(range(
                max(0, self.center_frame - self.num_frames // 2),
                self.center_frame + self.num_frames // 2
            ))
            
            for frame in container.decode(video_stream):
                if frame_count in target_frames:
                    img = frame.to_ndarray(format='rgb24')
                    frames.append(img)
                
                frame_count += 1
                if len(frames) >= self.num_frames:
                    break
            
            container.close()
            
            if not frames:
                return False, "No frames extracted", None
            
            frames_array = np.stack(frames, axis=0)
            return True, f"Success: {len(frames)} frames, resolution {frames_array.shape[1]}x{frames_array.shape[2]}", frames_array
            
        except Exception as e:
            return False, f"PyAV error: {str(e)}", None
    
    def test_opencv(self, video_path: Path) -> Tuple[bool, str, Optional[np.ndarray]]:
        """Test video loading with OpenCV."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                return False, "Could not open video with OpenCV", None
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                cap.release()
                return False, "No frames in video", None
            
            # Calculate frame indices
            start_frame = max(0, self.center_frame - self.num_frames // 2)
            end_frame = min(total_frames, start_frame + self.num_frames)
            
            frames = []
            for frame_idx in range(start_frame, end_frame):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                else:
                    break
            
            cap.release()
            
            if not frames:
                return False, "Could not read any frames", None
            
            frames_array = np.stack(frames, axis=0)
            return True, f"Success: {len(frames)} frames, resolution {frames_array.shape[1]}x{frames_array.shape[2]}", frames_array
            
        except Exception as e:
            return False, f"OpenCV error: {str(e)}", None
    
    def validate_video(self, video_path: Path) -> Dict:
        """Validate a single video file with all available methods."""
        self.results['total_videos'] += 1
        
        video_result = {
            'path': str(video_path),
            'size_mb': video_path.stat().st_size / (1024 * 1024),
            'methods': {}
        }
        
        # Test with available methods
        methods = [
            ('decord', self.test_decord),
            ('pyav', self.test_pyav),
            ('opencv', self.test_opencv)
        ]
        
        any_success = False
        
        for method_name, method_func in methods:
            success, message, frames = method_func(video_path)
            
            video_result['methods'][method_name] = {
                'success': success,
                'message': message,
                'available': method_name != 'decord' or DECORD_AVAILABLE,
                'frames_shape': frames.shape if frames is not None else None
            }
            
            if success:
                self.results['method_success'][method_name] += 1
                any_success = True
            else:
                self.results['method_failures'][method_name].append(str(video_path))
        
        if any_success:
            self.results['successful_videos'] += 1
        else:
            self.results['failed_videos'] += 1
            self.results['corrupted_videos'].append(str(video_path))
        
        video_result['any_success'] = any_success
        return video_result
    
    def validate_dataset(self, data_dir: Path, splits: List[str]) -> Dict:
        """Validate entire dataset."""
        all_results = []
        
        for split in splits:
            split_dir = data_dir / f"{split}_720p"
            
            if not split_dir.exists():
                logger.warning(f"Split directory {split_dir} does not exist, skipping")
                continue
            
            logger.info(f"Validating {split} split...")
            
            # Find all video files
            video_files = []
            for action_dir in split_dir.iterdir():
                if action_dir.is_dir() and action_dir.name.startswith('action_'):
                    video_files.extend(action_dir.glob("*.mp4"))
            
            logger.info(f"Found {len(video_files)} videos in {split} split")
            
            # Validate each video
            for i, video_path in enumerate(video_files):
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(video_files)} videos...")
                
                result = self.validate_video(video_path)
                result['split'] = split
                all_results.append(result)
        
        return {
            'summary': self.results,
            'detailed_results': all_results
        }


def print_summary(results: Dict):
    """Print validation summary."""
    summary = results['summary']
    
    print("\n" + "="*60)
    print("📊 DATASET VALIDATION SUMMARY")
    print("="*60)
    
    print(f"Total videos tested: {summary['total_videos']}")
    print(f"✅ Successful videos: {summary['successful_videos']}")
    print(f"❌ Failed videos: {summary['failed_videos']}")
    print(f"📈 Success rate: {summary['successful_videos']/summary['total_videos']*100:.1f}%")
    
    print(f"\n📋 Method Performance:")
    for method, count in summary['method_success'].items():
        available = method != 'decord' or DECORD_AVAILABLE
        if available:
            success_rate = count / summary['total_videos'] * 100
            print(f"  {method.upper()}: {count}/{summary['total_videos']} ({success_rate:.1f}%)")
        else:
            print(f"  {method.upper()}: Not available")
    
    if summary['corrupted_videos']:
        print(f"\n⚠️  CORRUPTED VIDEOS ({len(summary['corrupted_videos'])}):")
        for video in summary['corrupted_videos'][:10]:  # Show first 10
            print(f"  - {video}")
        if len(summary['corrupted_videos']) > 10:
            print(f"  ... and {len(summary['corrupted_videos']) - 10} more")


def save_results(results: Dict, output_dir: Path):
    """Save validation results to files."""
    output_dir.mkdir(exist_ok=True)
    
    # Save summary as JSON
    summary_path = output_dir / 'validation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results['summary'], f, indent=2)
    
    # Save detailed results as JSON
    detailed_path = output_dir / 'validation_detailed.json'
    with open(detailed_path, 'w') as f:
        json.dump(results['detailed_results'], f, indent=2)
    
    # Save corrupted videos list
    corrupted_path = output_dir / 'corrupted_videos.txt'
    with open(corrupted_path, 'w') as f:
        for video in results['summary']['corrupted_videos']:
            f.write(f"{video}\n")
    
    # Save CSV report
    csv_path = output_dir / 'validation_report.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Video Path', 'Split', 'Size (MB)', 'Any Success', 'Decord', 'PyAV', 'OpenCV'])
        
        for result in results['detailed_results']:
            writer.writerow([
                result['path'],
                result['split'],
                f"{result['size_mb']:.2f}",
                result['any_success'],
                result['methods'].get('decord', {}).get('success', False),
                result['methods'].get('pyav', {}).get('success', False),
                result['methods'].get('opencv', {}).get('success', False)
            ])
    
    logger.info(f"Results saved to {output_dir}")


def create_exclude_list(results: Dict, output_dir: Path):
    """Create a list of videos to exclude from training."""
    corrupted_videos = results['summary']['corrupted_videos']
    
    if not corrupted_videos:
        logger.info("No corrupted videos found, no exclude list needed")
        return
    
    exclude_path = output_dir / 'exclude_from_training.txt'
    with open(exclude_path, 'w') as f:
        f.write("# Videos to exclude from training due to corruption\n")
        f.write("# Add this list to your dataset loader to skip these files\n\n")
        for video in corrupted_videos:
            f.write(f"{video}\n")
    
    logger.info(f"Exclude list saved to {exclude_path}")


def suggest_solutions(results: Dict):
    """Suggest solutions based on validation results."""
    summary = results['summary']
    
    print("\n" + "="*60)
    print("💡 RECOMMENDED SOLUTIONS")
    print("="*60)
    
    if summary['failed_videos'] == 0:
        print("🎉 Great! All videos loaded successfully. No action needed.")
        return
    
    failure_rate = summary['failed_videos'] / summary['total_videos'] * 100
    
    if failure_rate < 5:
        print(f"✅ Low failure rate ({failure_rate:.1f}%). Recommended actions:")
        print("  1. Exclude corrupted videos from training (see exclude list)")
        print("  2. Continue training with current setup")
        
    elif failure_rate < 15:
        print(f"⚠️  Moderate failure rate ({failure_rate:.1f}%). Recommended actions:")
        print("  1. Try using PyAV instead of decord:")
        print("     pip install av")
        print("  2. Exclude corrupted videos from training")
        print("  3. Consider re-downloading problematic videos")
        
    else:
        print(f"🚨 High failure rate ({failure_rate:.1f}%). Recommended actions:")
        print("  1. Install alternative video loading libraries:")
        print("     pip install av  # PyAV")
        print("  2. Use OpenCV as fallback (slower but more compatible)")
        print("  3. Consider re-downloading the entire dataset")
        print("  4. Check if videos were corrupted during transfer")
    
    # Method-specific recommendations
    best_method = max(summary['method_success'], key=summary['method_success'].get)
    best_success = summary['method_success'][best_method]
    
    if best_success > summary['method_success'].get('decord', 0):
        print(f"\n💡 Best performing method: {best_method.upper()}")
        print(f"   Consider modifying dataset.py to prefer {best_method}")


def main():
    parser = argparse.ArgumentParser(description='Validate MVFouls dataset videos')
    parser.add_argument('--data-dir', type=str, default='mvfouls',
                       help='Path to mvfouls dataset directory')
    parser.add_argument('--splits', nargs='+', default=['train', 'test', 'valid'],
                       help='Dataset splits to validate')
    parser.add_argument('--output-dir', type=str, default='validation_results',
                       help='Directory to save validation results')
    parser.add_argument('--num-frames', type=int, default=32,
                       help='Number of frames to extract for testing')
    parser.add_argument('--center-frame', type=int, default=75,
                       help='Center frame for temporal window')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if not data_dir.exists():
        logger.error(f"Data directory {data_dir} does not exist")
        return
    
    logger.info(f"Starting dataset validation...")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Splits: {args.splits}")
    logger.info(f"Available methods: decord={DECORD_AVAILABLE}, PyAV={PYAV_AVAILABLE}, OpenCV=True")
    
    # Run validation
    validator = VideoValidator(args.num_frames, args.center_frame)
    start_time = time.time()
    
    results = validator.validate_dataset(data_dir, args.splits)
    
    end_time = time.time()
    logger.info(f"Validation completed in {end_time - start_time:.1f} seconds")
    
    # Print and save results
    print_summary(results)
    save_results(results, output_dir)
    create_exclude_list(results, output_dir)
    suggest_solutions(results)
    
    print(f"\n📁 All results saved to: {output_dir}")


if __name__ == "__main__":
    main() 