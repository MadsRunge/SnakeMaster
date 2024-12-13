import os
import subprocess
from pathlib import Path

def create_evolution_video(frames_dir: str = "training_run_recording2/recordings", 
                         output_file: str = "evolution.mp4", 
                         fps: int = 10):
    recordings_dir = Path(frames_dir)
    
    if not recordings_dir.exists():
        print(f"Directory not found: {recordings_dir}")
        return
        
    # Get all PNG files and sort them
    files = sorted([f for f in recordings_dir.iterdir() if f.suffix == '.png'])
    if not files:
        print("No PNG files found")
        return
        
    print(f"Found {len(files)} frames")
    
    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create temporary directory for renamed files
    temp_dir = recordings_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Rename files to sequential numbering
        for i, file in enumerate(files):
            new_name = temp_dir / f"frame_{i:06d}.png"
            os.symlink(file.absolute(), new_name)
        
        cmd = [
            'ffmpeg',
            '-y',
            '-framerate', str(fps),
            '-i', str(temp_dir / "frame_%06d.png"),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Video saved as: {output_file}")
        else:
            print("Error during generation:")
            print(result.stderr)
            
    finally:
        # Clean up temp directory
        for file in temp_dir.iterdir():
            file.unlink()
        temp_dir.rmdir()

if __name__ == "_main_":
    create_evolution_video()