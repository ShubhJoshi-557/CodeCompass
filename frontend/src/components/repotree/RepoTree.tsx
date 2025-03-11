"use client";

import { fetchLatestCommitSHA, fetchRepoTree } from "@/lib/githubapi";
import { useQuery } from "@tanstack/react-query";
import { useEffect } from "react";
import "react-folder-tree/dist/style.css";
import { toast } from "sonner";
import FileTree from "./FileExplorer";

interface RepoTreeProps {
  owner: string;
  repo: string;
  branch: string;
}

const RepoTree: React.FC<RepoTreeProps> = ({ owner, repo, branch }) => {
  // Fetch latest commit SHA
  const {
    data: commitSHA,
    error: fetchRepoError,
    isSuccess,
  } = useQuery({
    queryKey: ["commitSHA", owner, repo, branch],
    queryFn: () => fetchLatestCommitSHA({ owner, repo, branch }),
  });

  // Fetch repository tree
  const { data: repoTree } = useQuery({
    queryKey: ["repoTree", owner, repo, commitSHA],
    queryFn: () => fetchRepoTree({ owner, repo, commitSHA: commitSHA! }),
    enabled: !!commitSHA,
  });

  useEffect(() => {
    if (isSuccess) {
      toast.success("Repository fetched successfully", {
        // description: "Please add your Personal access token.",
        // action: {
        //   label: "Ok",
        //   onClick: () => console.log("ok"),
        // },
      });
    }
    if (fetchRepoError && fetchRepoError?.status === 401) {
      toast.warning("Invalid Token", {
        description: "Please add a valid Personal access token.",
        // action: {
        //   label: "Ok",
        //   onClick: () => console.log("Ok"),
        // },
      });
    } else if (fetchRepoError && fetchRepoError?.status === 403) {
      toast.warning("Rate Limit Exceeded", {
        // description: "Api rate limit exceeded.",
        // action: {
        //   label: "Ok",
        //   onClick: () => console.log("Ok"),
        // },
      });
    } else if (fetchRepoError && fetchRepoError?.status === 404) {
      toast.warning("Repository not found", {
        description: "Check the repo/branch name or add a personal access token for private repos.",
        // action: {
        //   label: "Ok",
        //   onClick: () => console.log("Ok"),
        // },
      });
    }
  }, [fetchRepoError, isSuccess, commitSHA, repoTree]);

  console.log(repoTree, fetchRepoError, commitSHA, "STATUS");
  if (!repoTree) return <p>Loading...</p>;

  return <div>{repoTree && <FileTree data={repoTree} />}</div>;
};

export default RepoTree;
