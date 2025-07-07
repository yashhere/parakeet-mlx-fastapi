from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class TranscriptionWord(BaseModel):
    """Word-level timestamp information."""

    word: str = Field(..., description="The text content of the word")
    start: float = Field(..., description="Start time of the word in seconds")
    end: float = Field(..., description="End time of the word in seconds")


class TranscriptionSegment(BaseModel):
    """Segment-level transcription information."""

    id: int = Field(..., description="Unique identifier of the segment")
    seek: int = Field(..., description="Seek offset in the audio")
    start: float = Field(..., description="Start time of the segment in seconds")
    end: float = Field(..., description="End time of the segment in seconds")
    text: str = Field(..., description="Text content of the segment")
    tokens: List[int] = Field(
        default_factory=list, description="Token IDs of the segment"
    )
    temperature: float = Field(
        default=0.0, description="Temperature used for generation"
    )
    avg_logprob: float = Field(
        default=0.0, description="Average log probability of the segment"
    )
    compression_ratio: float = Field(
        default=1.0, description="Compression ratio of the segment"
    )
    no_speech_prob: float = Field(default=0.0, description="Probability of no speech")


class TranscriptionUsage(BaseModel):
    """Usage information for the transcription."""

    type: Literal["duration"] = Field(
        default="duration", description="Type of usage measurement"
    )
    seconds: int = Field(..., description="Duration in seconds")


class TranscriptionResponse(BaseModel):
    """OpenAI-compatible transcription response."""

    task: Literal["transcribe"] = Field(default="transcribe", description="Task type")
    language: Optional[str] = Field(None, description="Detected or specified language")
    duration: Optional[float] = Field(
        None, description="Duration of the audio in seconds"
    )
    text: str = Field(..., description="The transcribed text")
    words: Optional[List[TranscriptionWord]] = Field(
        None, description="Word-level timestamps"
    )
    segments: Optional[List[TranscriptionSegment]] = Field(
        None, description="Segment-level information"
    )
    usage: Optional[TranscriptionUsage] = Field(None, description="Usage information")


# For different response formats
TranscriptionTextResponse = str
TranscriptionJsonResponse = TranscriptionResponse
