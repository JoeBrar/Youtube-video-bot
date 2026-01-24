"""
SRT File Utilities
Handles parsing of SRT subtitle files and grouping them into sentences for prompt generation.
"""
import re
from pathlib import Path
from datetime import timedelta

def parse_time(time_str):
    """Convert SRT time string (00:00:00,000) to seconds."""
    # Handle both comma and dot for milliseconds
    time_str = time_str.replace(',', '.')
    hours, minutes, seconds = time_str.split(':')
    seconds, milliseconds = seconds.split('.')
    
    total_seconds = (
        int(hours) * 3600 + 
        int(minutes) * 60 + 
        int(seconds) + 
        int(milliseconds) / 1000
    )
    return total_seconds

def parse_srt(file_path: Path):
    """
    Parse an SRT file into a list of subtitle objects.
    Returns: [{'index': 1, 'start': 1.0, 'end': 4.0, 'text': 'Hello world'}, ...]
    """
    if not file_path.exists():
        raise FileNotFoundError(f"SRT file not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by double newline to get blocks
    blocks = re.split(r'\n\s*\n', content.strip())
    subtitles = []

    for block in blocks:
        if not block.strip():
            continue
            
        lines = block.split('\n')
        if len(lines) < 3:
            continue

        try:
            # First line is index
            index = int(lines[0].strip())
            
            # Second line is timestamp
            timestamp_line = lines[1].strip()
            if '-->' not in timestamp_line:
                continue
                
            start_str, end_str = timestamp_line.split(' --> ')
            start_time = parse_time(start_str.strip())
            end_time = parse_time(end_str.strip())
            
            # Remaining lines are text
            text = ' '.join(lines[2:])
            # Clean up tags like <i> or <b>
            text = re.sub(r'<[^>]+>', '', text)
            
            subtitles.append({
                'index': index,
                'start': start_time,
                'end': end_time,
                'text': text.strip()
            })
        except ValueError:
            continue

    return subtitles

def group_srt_by_sentences(subtitles):
    """
    Group subtitles into full sentences or meaningful segments.
    Returns list of dicts:
    [{'text': 'Full sentence.', 'start': 0.0, 'end': 5.5, 'subtitles': [...]}, ...]
    """
    segments = []
    current_segment = {
        'text': '',
        'start': None,
        'end': None,
        'subtitles': []
    }

    for sub in subtitles:
        text = sub['text']
        
        # Initialize start time if needed
        if current_segment['start'] is None:
            current_segment['start'] = sub['start']
        
        # Space handling
        prefix = " " if current_segment['text'] and not current_segment['text'].endswith(" ") else ""
        current_segment['text'] += prefix + text
        
        # Always update end time
        current_segment['end'] = sub['end']
        current_segment['subtitles'].append(sub)

        # Check for sentence endings (. ? !)
        # Ensure it's not an abbreviation like Mr. or Dr. (basic check)
        if re.search(r'[.?!](?:["\']|)$', text):
            # Basic heuristic: if it ends with punctuation, close the segment
            segments.append(current_segment)
            current_segment = {
                'text': '',
                'start': None,
                'end': None,
                'subtitles': []
            }

    # Add any remaining text
    if current_segment['text']:
        segments.append(current_segment)

    return segments
