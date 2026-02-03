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
    Group subtitles into full sentences, splitting mid-subtitle if necessary.
    Returns list of dicts:
    [{'text': 'Full sentence.', 'start': 0.0, 'end': 5.5}, ...]
    """
    segments = []
    current_segment = {
        'text': '',
        'start': None,
        'end': None
    }

    # Regex for sentence ending: period, question mark, or exclamation mark
    # followed by a space and a capital letter, or end of string.
    # We also handle quotes.
    # This is a basic split; for more robustness we'd need NLTK but we want to avoid deps.
    # We'll stick to a simple split on punctuation followed by space/end for now.
    
    def get_sentence_split_index(text):
        """Find the index where a sentence likely ends."""
        # Check for punctuation markers
        matches = list(re.finditer(r'([.?!]["\']?)(\s+|$)', text))
        for match in matches:
            # Basic heuristic: ignore common abbreviations if simple logic needed
            # For now, we assume standard punctuation is a split
            # We return the END of the sentence (including punctuation)
            return match.end(1)
        return -1

    for sub in subtitles:
        text = sub['text']
        start_time = sub['start']
        end_time = sub['end']
        duration = end_time - start_time
        
        # Process the text of this subtitle
        while text:
            # Initialize start time for the current segment if it's new
            if current_segment['start'] is None:
                current_segment['start'] = start_time
            
            split_idx = get_sentence_split_index(text)
            
            if split_idx != -1:
                # Found a sentence ending in this subtitle block
                sentence_part = text[:split_idx]
                remaining_text = text[split_idx:].strip()
                
                # Append this part to the current segment
                prefix = " " if current_segment['text'] and not current_segment['text'].endswith(" ") else ""
                current_segment['text'] += prefix + sentence_part
                
                # Calculate the end time for this sentence
                # Proportional to the length of text consumed from this subtitle
                total_char_len = len(sub['text']) # Original length
                # We need to approximate how much time this part took
                # Note: this is an estimation. 
                # If we are splitting "Hello. World", "Hello." took some % of the time.
                # However we might have already consumed some of sub['text'] in previous loop?
                # Actually, let's keep it simple: assume constant speaking rate within the subtitle block.
                
                # We need to know how much of the CURRENT 'text' variable (which shrinks) corresponds to time
                # But 'text' is just a slice. We should use the ratio of (sentence_part / original_text_len) * duration
                # But we might have multiple sentences. 
                
                # Let's track consumed time within this subtitle
                # Or simpler: re-calculate ratio based on char counts of the parts we extract
                
                consumed_ratio = len(sentence_part) / len(text) if len(text) > 0 else 1
                segment_duration = duration * consumed_ratio
                
                # The end time of this sentence is start_time + segment_duration
                segment_end_time = start_time + segment_duration
                current_segment['end'] = segment_end_time
                
                # Finalize the segment
                segments.append(current_segment)
                
                # Reset for next segment
                current_segment = {
                    'text': '',
                    'start': None,  # Will be set in next iteration
                    'end': None
                }
                
                # Update loop variables for the remaining text
                text = remaining_text
                start_time = segment_end_time # Next sentence starts where this one ended
                duration = end_time - start_time # Remaining duration
                
                if duration < 0: duration = 0 # Safety
                
            else:
                # No sentence split found, just append the rest
                prefix = " " if current_segment['text'] and not current_segment['text'].endswith(" ") else ""
                current_segment['text'] += prefix + text
                
                # Update the end time of the current segment to be the end of this subtitle
                current_segment['end'] = end_time
                text = "" # Done with this subtitle

    # Add any remaining text
    if current_segment['text']:
        segments.append(current_segment)

    return segments

