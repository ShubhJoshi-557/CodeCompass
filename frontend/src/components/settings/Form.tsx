"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { z } from "zod";

import { Button } from "@/components/ui/button";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import useStore from "@/store/store";
import { fetchLatestCommitSHA, fetchRepoTree } from "@/lib/githubapi";
import { toast } from "sonner";
import { useMutation, useQueryClient } from "@tanstack/react-query";

const formSchema = z.object({
  repo_url: z
    .string()
    .regex(
      /^https:\/\/(?:[^@]+@)?github\.com\/([^\/]+)\/([^\/]+)(?:\.git)?(?:\/tree\/([^\/]+))?$/,
      "Invalid GitHub repository URL"
    ),
  branch: z
    .string()
    .regex(/^(?!.*[\/.]{2})[a-zA-Z0-9._-]+$/, "Invalid GitHub branch name"),
  token: z.union([
    z
      .string()
      .regex(
        /^gh[pous]_[A-Za-z0-9_]{36,255}$/,
        "Invalid GitHub Personal Access Token"
      ),
    z.literal(""),
    z.undefined(),
  ]),
});

export function RepoForm({ closeDialog }: { closeDialog: () => void }) {
  const { updateCurrentRepo, setCurrentFolder, repoTreeLoading, setRepoTreeLoading } = useStore();
  const queryClient = useQueryClient();

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      repo_url: "",
      branch: "",
      token: "",
    },
  });
 
  function parseGitHubURL(values: any) {
    const regex =
      /github\.com\/([^\/]+)\/([^\/]+)(?:\.git)?(?:\/tree\/([^\/]+))?/;
    const match = values.repo_url.match(regex);
    if (match) {
      return {
        link: values.repo_url,
        owner: match[1],
        repo: match[2].split(".git")[0],
        branch: values.branch || match[3] || "default (main/master)",
        token: values.token,
      };
    }
    return null;
  }

  const fetchSHA = useMutation({
    mutationFn: async ({ owner, repo, branch }: any) =>
      fetchLatestCommitSHA({ owner, repo, branch }),
  });

  const fetchTree = useMutation({
    mutationFn: async ({ owner, repo, commitSHA }: any) =>
      fetchRepoTree({ owner, repo, commitSHA }),
  });

  async function onSubmit(values: z.infer<typeof formSchema>) {
    console.log("Form submitted:", values);

    const repo_obj = parseGitHubURL(values);
    if (!repo_obj) {
      toast.error("Invalid GitHub repository URL.");
      return;
    }

    setRepoTreeLoading(true); // Start loading state

    try {
      const commitSHA = await fetchSHA.mutateAsync({
        owner: repo_obj.owner,
        repo: repo_obj.repo,
        branch: repo_obj.branch,
      });

      if (!commitSHA) {
        toast.error("Repository not fetched!", {
          description: `Repository size exceeds limit (50MB). Skipping fetch.`,
        });
        setRepoTreeLoading(false);
        return;
      }

      const repoTree = await fetchTree.mutateAsync({
        owner: repo_obj.owner,
        repo: repo_obj.repo,
        commitSHA,
      });

      if (!repoTree || repoTree.length === 0) {
        toast.error("Failed to fetch repository tree.");
        setRepoTreeLoading(false);
        return;
      }

      setCurrentFolder("");

      // ✅ Update Zustand store only if successful
      updateCurrentRepo("link", repo_obj.link);
      updateCurrentRepo("owner", repo_obj.owner);
      updateCurrentRepo("repo", repo_obj.repo);
      updateCurrentRepo("branch", repo_obj.branch);
      updateCurrentRepo("token", repo_obj.token);
      updateCurrentRepo("repoTree", repoTree);

      toast.success("Repository fetched successfully");
      closeDialog();
      queryClient.invalidateQueries({ queryKey: ["repoTree"] });
    } catch (error: any) {
      if (error.response) {
        const status = error.response.status;
        if (status === 401) {
          toast.warning("Invalid Token", {
            description: "Please add a valid Personal Access Token.",
          });
        } else if (status === 403) {
          toast.warning("Rate Limit Exceeded");
        } else if (status === 404) {
          toast.warning("Repository not found", {
            description:
              "Check the repo/branch name or add a personal access token for private repos.",
          });
        } else {
          toast.error("Failed to fetch repository", {
            description: `Unexpected error: ${status}`,
          });
        }
      } else {
        console.error("Unknown error:", error);
        toast.error("An unexpected error occurred.");
      }
    } finally {
      setRepoTreeLoading(false); // Stop loading state
    }
  }

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
        <FormField
          control={form.control}
          name="repo_url"
          render={({ field }) => (
            <FormItem>
              <div className="grid grid-cols-4 items-center gap-4">
                <FormLabel>Repository</FormLabel>
                <FormControl className="col-span-3">
                  <Input
                    placeholder="eg: https://github.com/ShubhJoshi-557/CodeCompass.git"
                    {...field}
                  />
                </FormControl>
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <FormLabel> </FormLabel>
                <FormMessage className="col-span-3" />
              </div>
            </FormItem>
          )}
        />
        <FormField
          control={form.control}
          name="branch"
          render={({ field }) => (
            <FormItem>
              <div className="grid grid-cols-4 items-center gap-4">
                <FormLabel>Branch</FormLabel>
                <FormControl className="col-span-3">
                  <Input placeholder="eg: main" {...field} />
                </FormControl>
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <FormLabel> </FormLabel>
                <FormMessage className="col-span-3" />
              </div>
            </FormItem>
          )}
        />
        <FormField
          control={form.control}
          name="token"
          render={({ field }) => (
            <FormItem>
              <div className="grid grid-cols-4 items-center gap-4">
                <FormLabel>Token</FormLabel>
                <FormControl className="col-span-3">
                  <Input
                    placeholder="eg: ghp_1234567890abcdefghijklmnopqrstuvwxyzABCD"
                    {...field}
                  />
                </FormControl>
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <FormLabel> </FormLabel>
                <FormDescription className="col-span-3">
                  Only required for private repositories.
                </FormDescription>
              </div>
              <div className="grid grid-cols-4 items-center gap-4">
                <FormLabel> </FormLabel>
                <FormMessage className="col-span-3" />
              </div>
            </FormItem>
          )}
        />
          <Button className="cursor-pointer" type="submit" disabled={repoTreeLoading}>
            {repoTreeLoading ? "Fetching..." : "Save"}
          </Button>
        
        
      </form>
    </Form>
  );
}
