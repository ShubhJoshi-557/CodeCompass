import { useState, useEffect } from "react";

const Sidebar = ({ onSelectFile }) => {
  const [repoStructure, setRepoStructure] = useState([]);
  const [expandedFolders, setExpandedFolders] = useState({});

  useEffect(() => {
    fetchRepoStructure();
  }, []);

  const fetchRepoStructure = async () => {
    const response = await fetch(
      "https://api.github.com/repos/{owner}/{repo}/contents/"
    );
    const data = await response.json();
    setRepoStructure(data);
  };

  const toggleFolder = (folderName) => {
    setExpandedFolders((prev) => ({
      ...prev,
      [folderName]: !prev[folderName],
    }));
  };

  return (
    <div className="w-64 bg-gray-900 text-white h-full p-4 overflow-y-auto">
      <h2 className="text-lg font-bold mb-4">📂 CodeCompass</h2>
      <ul>
        {repoStructure.map((item) => (
          <li key={item.name}>
            {item.type === "dir" ? (
              <div>
                <button onClick={() => toggleFolder(item.name)}>
                  {expandedFolders[item.name] ? "📂" : "📁"} {item.name}
                </button>
                {expandedFolders[item.name] && <SubFolder path={item.path} onSelectFile={onSelectFile} />}
              </div>
            ) : (
              <div className="ml-4 cursor-pointer" onClick={() => onSelectFile(item.path)}>
                📄 {item.name}
              </div>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
};

const SubFolder = ({ path, onSelectFile }) => {
  const [files, setFiles] = useState([]);

  useEffect(() => {
    fetch(`https://api.github.com/repos/{owner}/{repo}/contents/${path}`)
      .then((res) => res.json())
      .then((data) => setFiles(data));
  }, [path]);

  return (
    <ul className="ml-4">
      {files.map((file) => (
        <li key={file.name} className="cursor-pointer" onClick={() => file.type === "file" && onSelectFile(file.path)}>
          {file.type === "dir" ? `📂 ${file.name}` : `📄 ${file.name}`}
        </li>
      ))}
    </ul>
  );
};

export default Sidebar;