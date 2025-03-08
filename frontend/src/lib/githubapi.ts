import axios from "axios";
import useStore from "../store/store"; // Import Zustand store

const GITHUB_API_URL = "https://api.github.com/repos";

// Function to get headers dynamically
const getHeaders = () => {
  const token = useStore.getState().currentRepo.token; // Fetch token from Zustand store
  return {
    Authorization: token ? `token ${token}` : "", // Only add token if available
    Accept: "application/vnd.github.v3+json",
  };
};

// Define types
export interface GitHubTreeItem {
  path: string;
  type: "blob" | "tree"; // "blob" = file, "tree" = folder
}

export interface FetchCommitSHAParams {
  owner: string;
  repo: string;
  branch: string;
}

export interface FetchRepoTreeParams {
  owner: string;
  repo: string;
  commitSHA: string;
}

// Fetch latest commit SHA of the branch
export const fetchLatestCommitSHA = async ({
  owner,
  repo,
  branch,
}: FetchCommitSHAParams): Promise<string> => {
  const response = await axios.get(
    `${GITHUB_API_URL}/${owner}/${repo}/branches/${branch}`,
    { headers: getHeaders() }
  );
    console.log(response, "RESSS")
  return response.data.commit.sha;
};

// Fetch the repository tree
export const fetchRepoTree = async ({
  owner,
  repo,
  commitSHA,
}: FetchRepoTreeParams): Promise<GitHubTreeItem[]> => {
  const response = await axios.get(
    `${GITHUB_API_URL}/${owner}/${repo}/git/trees/${commitSHA}?recursive=1`,
    { headers: getHeaders() }
  );
  return response.data.tree;
};
