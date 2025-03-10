from pydantic import BaseModel

class RepoRequest(BaseModel):
    owner: str
    repo: str
    branch: str
    token: str  # GitHub Personal Access Token (PAT)
