"use client";

import { Cormorant_Garamond } from "next/font/google";
import { useState } from "react";

const garamond = Cormorant_Garamond({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

export default function Home() {
  const [showFiles, setShowFiles] = useState(false);
  const [selectedFileName, setSelectedFileName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [classification, setClassification] = useState<string | null>(null);
  const [confidence, setConfidence] = useState<number | string | null>(null);
  const [trueLabel, setTrueLabel] = useState<string | number | null>(null);

  let csvFiles = [];
  for (let i = 0; i < 20; i++) {
    csvFiles.push(`fraudulent_test_${i + 1}.csv`);
    csvFiles.push(`legitimate_test_${i + 1}.csv`);
  }

  const chooseCsvFile = async (fileName: string) => {
    const response = await fetch(`/transactions/${fileName}`);

    if (!response.ok) {
      console.error("Failed to load CSV file.");
      return;
    }

    const blob = await response.blob();

    const selectedFile = new File([blob], fileName, {
      type: "text/csv",
    });

    setFile(selectedFile);
    setSelectedFileName(fileName);
    setShowFiles(false);
  };

  const display_files = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    setShowFiles(!showFiles);
  };

  const predict = async (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();

    if (!file) {
      console.error("File has not been selected.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to make prediction.");
      }

      const data = await response.json();

      await recordTransaction(data);

      setClassification(data.classification);
      setConfidence(data.confidence);
      setTrueLabel(data.true_label);
    } catch (error) {
      console.error("Failed to analyse the transaction: ", error);
    }
  };

  const recordTransaction = async (data : JSON) => {
    const response = await fetch("http://127.0.0.1:8000/record-transaction", {
      method: "POST",
      headers: {
        "content-type": "application/json"
      },
      body: JSON.stringify(data)
    })

    if (!response.ok) {
      console.error("Failed to record transaction.");
      return;
    }

    console.log("Successfully recorded transaction.")
  }

  return (
  <main className="min-h-screen bg-[#f3eadc] px-10 py-6 text-[#211a16]">
    <div className="mx-auto max-w-6xl">
      {/* Navbar */}
      <nav className="border-t-4 border-[#c8ad78] border-b border-[#c8b18a] py-3">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <span
            className={`${garamond.className} text-xl font-semibold tracking-tight text-[#6b1f2a]`}
          >
            Sentry
          </span>

          <div className="flex flex-wrap gap-x-7 gap-y-2 text-[11px] font-semibold uppercase tracking-[0.22em] text-[#6d5a43]">
            <a href="" className="transition hover:text-[#6b1f2a]">
              Detector
            </a>

            <a href="" className="transition hover:text-[#6b1f2a]">
              Model Performance
            </a>

            <a href="" className="transition hover:text-[#6b1f2a]">
              Dataset
            </a>

            <a href="" className="transition hover:text-[#6b1f2a]">
              About Us
            </a>
          </div>
        </div>
      </nav>

      {/* Header */}
      <header className="border-b border-[#c8b18a] py-8">
        <h1
          className={`${garamond.className} text-6xl font-semibold tracking-tight text-[#6b1f2a] md:text-7xl`}
        >
          Detector
        </h1>

        <p
          className={`${garamond.className} mt-2 text-xl text-[#6d5a43] md:text-2xl`}
        >
          Fraudulent Ethereum Transaction Detector
        </p>
      </header>

      {/* Main detector section */}
      <section className="mt-8 grid border border-[#c8b18a] bg-[#fffaf2] lg:grid-cols-[1.15fr_0.85fr]">
        <section className="border-b border-[#c8b18a] p-6 lg:border-b-0 lg:border-r">
          <div className="border-b border-[#d8c6aa] pb-5">
            <p className="text-[11px] font-semibold uppercase tracking-[0.28em] text-[#8a6b3d]">
              Secure ML Review
            </p>

            <h2
              className={`${garamond.className} mt-3 text-3xl font-semibold text-[#2b211b] md:text-4xl`}
            >
              Review a Transaction
            </h2>

            <p className="mt-4 max-w-2xl text-sm leading-6 text-[#67594d]">
              Select a prepared CSV transaction file and submit it for model
              analysis. Sentry will return its classification, confidence level,
              and the known label for comparison.
            </p>

            <p className="pt-4 text-sm leading-6 text-[#67594d]">
              The model has not seen these transactions previously, nor will it
              memorise these transactions.
            </p>
          </div>

          <form className="mt-6 space-y-5">
            <div className="flex flex-col gap-3 sm:flex-row sm:flex-wrap">
              <button
                type="button"
                onClick={display_files}
                className="border border-[#b89d70] bg-[#fffaf2] px-4 py-2.5 text-xs font-semibold uppercase tracking-wide text-[#3a2d24] transition hover:bg-[#efe2cf]"
              >
                {selectedFileName ? "Change Transaction File" : "Select a File"}
              </button>

              <button
                type="button"
                onClick={predict}
                className="border border-[#6b1f2a] bg-[#6b1f2a] px-4 py-2.5 text-xs font-semibold uppercase tracking-wide text-white transition hover:bg-[#4f1720]"
              >
                Analyse Transaction
              </button>
            </div>

            {showFiles && (
              <div className="border border-[#c8b18a] bg-[#f8f0e4] p-4">
                <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-[#8a6b3d]">
                  Available Files
                </p>

                <div className="mt-3 grid gap-2">
                  {csvFiles.map((filename) => (
                    <button
                      key={filename}
                      type="button"
                      onClick={() => chooseCsvFile(filename)}
                      className="border border-[#c8b18a] bg-[#fffaf2] px-3 py-2.5 text-left text-xs font-medium text-[#3a2d24] transition hover:bg-[#efe2cf]"
                    >
                      {filename}
                    </button>
                  ))}
                </div>
              </div>
            )}

            <div className="border border-[#d8c6aa] bg-[#fffdf8] p-4">
              <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-[#8a6b3d]">
                Current Selection
              </p>

              <p className="mt-2 text-sm text-[#3a2d24]">
                {selectedFileName || "No CSV file selected yet."}
              </p>
            </div>
          </form>
        </section>

        {/* Prediction output */}
        <aside className="bg-[#fbf4e9] p-6">
          <div className="border-b border-[#c8b18a] pb-5">
            <p className="text-[11px] font-semibold uppercase tracking-[0.28em] text-[#8a6b3d]">
              Prediction Output
            </p>

            <h3
              className={`${garamond.className} mt-3 text-3xl font-semibold text-[#2b211b] md:text-4xl`}
            >
              Model Decision
            </h3>

            <p className="mt-4 text-sm leading-6 text-[#67594d]">
              The model will output its decision regarding the selected
              transaction.
            </p>
          </div>

          {classification !== null ? (
            <div className="mt-6 divide-y divide-[#d8c6aa] border border-[#c8b18a] bg-[#fffaf2]">
              <div className="p-4">
                <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-[#8a6b3d]">
                  Classification
                </p>

                <p
                  className={`${garamond.className} mt-2 text-3xl font-semibold text-[#6b1f2a]`}
                >
                  {classification}
                </p>
              </div>

              <div className="p-4">
                <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-[#8a6b3d]">
                  Confidence
                </p>

                <p
                  className={`${garamond.className} mt-2 text-3xl font-semibold text-[#2b211b]`}
                >
                  {confidence}%
                </p>
              </div>

              <div className="p-4">
                <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-[#8a6b3d]">
                  True Label
                </p>

                <p
                  className={`${garamond.className} mt-2 text-3xl font-semibold text-[#2b211b]`}
                >
                  {trueLabel}
                </p>
              </div>
            </div>
          ) : (
            <div className="mt-6 border border-dashed border-[#b89d70] bg-[#fffaf2] p-5">
              <p className="text-sm leading-6 text-[#67594d]">
                No analysis has been run yet. Select a transaction file, then
                click{" "}
                <span className="font-semibold text-[#6b1f2a]">
                  Analyse Transaction
                </span>
                .
              </p>
            </div>
          )}
        </aside>
      </section>
    </div>
  </main>
  );
}