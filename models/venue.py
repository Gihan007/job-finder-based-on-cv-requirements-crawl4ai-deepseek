from pydantic import BaseModel


class Venue(BaseModel):
    title: str
    company: str
    location: str
    employment_type: str
    required_skills: str
    experience_level: str
    match_reason: str
