import React, { useEffect, useState } from "react";
import { initializeModel, processImage } from "./lib/process";

export default function App() {
  const [image, setImage] = useState<{
    id: number;
    file: File;
    maskFile?: File;
    processedFile?: File;
  } | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const initialized = await initializeModel();
        if (!initialized) {
          throw new Error("Failed to initialize background removal model");
        }
      } catch (err) {
        console.error(err);
      }
      setIsLoading(false);
    })();
  }, []);

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files[0]) {
      const newImage = {
        id: Date.now(),
        file: event.target.files[0],
        processedFile: undefined,
      };
      setImage(newImage);
    }
  };

  const handleClick = async (image: {
    id: number;
    file: File;
    processedFile?: File;
  }) => {
    try {
      const result = await processImage(image.file);
      if (result) {
        const { maskFile, processedFile } = result;
        setImage({ ...image, processedFile, maskFile });
      }
    } catch (error) {
      console.error("Error processing image:", error);
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-white text-gray-800 flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-600 mb-4"></div>
          <p className="text-lg">Loading background removal model...</p>
        </div>
      </div>
    );
  }
  return (
    <div className="h-screen w-full ">
      <div className="flex justify-center items-center h-20">
        <h1>Upload an Image</h1>
        <input type="file" accept="image/*" onChange={handleImageUpload} />
      </div>
      <div className="flex px-2 justify-center">
        <div className="max-w-md aspect-auto">
          {image && (
            <img src={URL.createObjectURL(image.file)} alt="Uploaded" />
          )}
        </div>
        <div className="max-w-md aspect-auto">
          {image?.maskFile && (
            <img src={URL.createObjectURL(image.maskFile)} alt="Processed" />
          )}
        </div>
        <div className="max-w-md aspect-auto">
          {image?.processedFile && (
            <img
              src={URL.createObjectURL(image.processedFile)}
              alt="Processed"
            />
          )}
        </div>
      </div>

      {image && (
        <div className="w-full flex justify-center mt-4">
          <button
            onClick={() => handleClick(image)}
            className="border-2 p-1 rounded-md"
          >
            Process Image
          </button>
        </div>
      )}
    </div>
  );
}
