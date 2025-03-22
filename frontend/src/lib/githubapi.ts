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

// ✅ Fetch repository metadata (to get size)
export const fetchRepoMetadata = async (owner: string, repo: string): Promise<number> => {
  const response = await axios.get(`${GITHUB_API_URL}/${owner}/${repo}`, {
    headers: getHeaders(),
  });
  return response.data.size; // Size in KB
};

// ✅ Fetch latest commit SHA (with conditional size check)
export const fetchLatestCommitSHA = async ({
  owner,
  repo,
  branch,
}: FetchCommitSHAParams): Promise<string | null> => {
  const isDemo = process.env.NEXT_PUBLIC_VERSION === "DEMO";

  if (isDemo) {
    const size = await fetchRepoMetadata(owner, repo);
    if (size > 50000) {
      console.warn(`Repository size (${size / 1024} MB) exceeds limit (50MB). Skipping fetch.`);
      return null;
    } 
  }

  const response = await axios.get(
    `${GITHUB_API_URL}/${owner}/${repo}/branches/${branch}`,
    { headers: getHeaders() }
  );
  return response.data.commit.sha;
};

// ✅ Fetch repo tree (only if repo size is ≤ 50MB in DEMO mode)
export const fetchRepoTree = async ({
  owner,
  repo,
  commitSHA,
}: FetchRepoTreeParams): Promise<GitHubTreeItem[]> => {
  if (!commitSHA) {
    console.warn("Skipping repo tree fetch: No commit SHA available.");
    return [];
  }

  const response = await axios.get(
    `${GITHUB_API_URL}/${owner}/${repo}/git/trees/${commitSHA}?recursive=1`,
    { headers: getHeaders() }
  );
  return response.data.tree;
};
