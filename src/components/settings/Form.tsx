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

export function ProfileForm() {
  const { currentRepo, updateCurrentRepo } = useStore();
  // 1. Define your form.
  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      repo_url: "",
      branch: "",
      token: "",
    },
  });

  function parseGitHubURL(values) {
    const regex =
      /github\.com\/([^\/]+)\/([^\/]+)(?:\.git)?(?:\/tree\/([^\/]+))?/;
    const match = values.repo_url.match(regex);
    if (match) {
      return {
        link: values.repo_url,
        owner: match[1],
        repo: match[2].split(".git")[0],
        branch: values.branch || match[3] || "default (main/master)",
        token:
          values.token,
      };
    } else {
      return null;
    }
  }
  // 2. Define a submit handler.
  function onSubmit(values: z.infer<typeof formSchema>) {
    // Do something with the form values.
    // ✅ This will be type-safe and validated.
    console.log(values);
    const repo_obj = parseGitHubURL(values);
    if (repo_obj) {
      updateCurrentRepo("link", repo_obj.link);
      updateCurrentRepo("owner", repo_obj.owner);
      updateCurrentRepo("repo", repo_obj.repo);
      updateCurrentRepo("branch", repo_obj.branch);
      updateCurrentRepo("token", repo_obj.token);

      console.log("Updated currentRepo:", repo_obj);
    } else {
      console.error("Invalid GitHub repository URL.");
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
        <Button type="submit">Save</Button>
      </form>
    </Form>
  );
}
