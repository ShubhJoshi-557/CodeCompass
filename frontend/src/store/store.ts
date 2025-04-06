import { create } from "zustand";
import { persist } from "zustand/middleware";

// 📌 Define File Type
interface FileItem {
  id: string;
  name: string;
  path: string;
  isFolder: boolean;
  children?: FileItem[];
}

// 📌 Define Repository Type
interface Repo {
  link: string;
  repo: string;
  owner: string;
  branch: string;
  token: string;
  repoTree: [];
}

// 📌 Define Store Interface
interface CodeCompassState {
  files: FileItem[];
  searchResults: FileItem[];
  explanations: Record<string, string>;
  currentRepo: Repo;
  currentFolder: string;
  repoTreeLoading: boolean;

  setFiles: (files: FileItem[]) => void;
  setSearchResults: (results: FileItem[]) => void;
  setExplanation: (filePath: string, explanation: string) => void;
  setCurrentRepo: (repo: Repo) => void;
  updateCurrentRepo: (key: keyof Repo, value: string | any) => void;
  setCurrentFolder: (currentFolder: string) => void;
  setRepoTreeLoading: (repoTreeLoading: boolean) => void;
}

// 📌 Create Zustand Store
const useStore = create<CodeCompassState>()(
  persist(
    (set) => ({
      files: [],
      searchResults: [],
      explanations: {},
      repoTreeLoading: false,

      // 🏷️ Default repo state
      currentRepo: {
        link: "",
        repo: "",
        owner: "",
        branch: "",
        token: "",
        repoTree: [],
      },

      currentFolder: "",
      // 🔍 Setters
      setFiles: (files) => set({ files }),
      setSearchResults: (results) => set({ searchResults: results }),
      setRepoTreeLoading: (repoTreeLoading) => set({ repoTreeLoading }),
      setExplanation: (filePath, explanation) =>
        set((state) => ({
          explanations: { ...state.explanations, [filePath]: explanation },
        })),

      // 🔄 Set Entire Repo
      setCurrentRepo: (repo) => set({ currentRepo: repo }),
      setCurrentFolder: (currentFolder) => set({ currentFolder }),
      // 🔄 Update Specific Repo Property
      updateCurrentRepo: (key, value) =>
        set((state) => ({
          currentRepo: { ...state.currentRepo, [key]: value },
        })),
    }),
    {
      name: "codecompass-storage", // Persistent state
    }
  )
);

export default useStore;
