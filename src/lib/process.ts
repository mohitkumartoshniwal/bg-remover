import {
  env,
  AutoModel,
  AutoProcessor,
  RawImage,
  PreTrainedModel,
  Processor,
} from "@huggingface/transformers";

const MODEL_ID = "briaai/RMBG-1.4";

interface ModelState {
  model: PreTrainedModel | null;
  processor: Processor | null;
  currentModelId: string;
}

const state: ModelState = {
  model: null,
  processor: null,
  currentModelId: MODEL_ID,
};

// Initialize the model based on the selected model ID
export async function initializeModel(): Promise<boolean> {
  try {
    env.allowLocalModels = false;
    if (env.backends?.onnx?.wasm) {
      env.backends.onnx.wasm.proxy = true;
    }

    state.model = await AutoModel.from_pretrained(MODEL_ID, {
      config: {
        model_type: "custom",
        is_encoder_decoder: false,
        max_position_embeddings: 0,
        "transformers.js_config": {
          kv_cache_dtype: undefined,
          free_dimension_overrides: undefined,
          device: undefined,
          dtype: undefined,
          use_external_data_format: undefined,
        },
        normalized_config: undefined,
      },
    });

    state.processor = await AutoProcessor.from_pretrained(MODEL_ID, {
      config: {
        do_normalize: true,
        do_pad: false,
        do_rescale: true,
        do_resize: true,
        image_mean: [0.5, 0.5, 0.5],
        feature_extractor_type: "ImageFeatureExtractor",
        image_std: [1, 1, 1],
        resample: 2,
        rescale_factor: 1 / 255,
        size: { width: 1024, height: 1024 },
      },
    });

    state.currentModelId = MODEL_ID;
    return true;
  } catch (error) {
    console.error("Error initializing model:", error);
    throw new Error(
      error instanceof Error
        ? error.message
        : "Failed to initialize background removal model"
    );
  }
}

export async function processImage(image: File) {
  if (!state.model || !state.processor) {
    throw new Error("Model not initialized. Call initializeModel() first.");
  }

  const img = await RawImage.fromURL(URL.createObjectURL(image));

  try {
    // Pre-process image
    const { pixel_values } = await state.processor(img);

    // Predict alpha matte
    const { output } = await state.model({ input: pixel_values });

    // Resize mask back to original size
    const maskData = (
      await RawImage.fromTensor(output[0].mul(255).to("uint8")).resize(
        img.width,
        img.height
      )
    ).data;

    // create mask canvas
    const maskCanvas = document.createElement("canvas");
    maskCanvas.width = img.width;
    maskCanvas.height = img.height;
    const maskCtx = maskCanvas.getContext("2d");
    if (!maskCtx) throw new Error("Could not get 2d context");

    // Draw mask data to mask canvas
    const maskPixelData = maskCtx.createImageData(img.width, img.height);
    for (let i = 0; i < maskData.length; ++i) {
      const value = maskData[i]; // grayscale value
      maskPixelData.data[4 * i] = value;
      maskPixelData.data[4 * i + 1] = value;
      maskPixelData.data[4 * i + 2] = value;
      maskPixelData.data[4 * i + 3] = 255;
    }

    maskCtx.putImageData(maskPixelData, 0, 0);

    // Convert mask canvas to blob
    const maskBlob = await new Promise<Blob>((resolve, reject) =>
      maskCanvas.toBlob(
        (blob) =>
          blob ? resolve(blob) : reject(new Error("Failed to create blob")),
        "image/png"
      )
    );

    // Create new image object with mask blob
    const maskFileName = `${image.name.split(".")[0]}-mask.png`;
    const maskFile = new File([maskBlob], maskFileName, {
      type: "image/png",
    });

    // Create new canvas
    const canvas = document.createElement("canvas");
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Could not get 2d context");

    // Draw original image output to canvas
    ctx.drawImage(img.toCanvas(), 0, 0);

    // Update alpha channel
    const pixelData = ctx.getImageData(0, 0, img.width, img.height);
    for (let i = 0; i < maskData.length; ++i) {
      pixelData.data[4 * i + 3] = maskData[i];
    }
    ctx.putImageData(pixelData, 0, 0);

    // Convert canvas to blob
    const blob = await new Promise<Blob>((resolve, reject) =>
      canvas.toBlob(
        (blob) =>
          blob ? resolve(blob) : reject(new Error("Failed to create blob")),
        "image/png"
      )
    );

    const [fileName] = image.name.split(".");
    const processedFile = new File([blob], `${fileName}-bg-blasted.png`, {
      type: "image/png",
    });
    return { maskFile, processedFile };
  } catch (error) {
    console.error("Error processing image:", error);
    throw new Error("Failed to process image");
  }
}
