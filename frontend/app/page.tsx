"use client"

import { Cormorant_Garamond } from "next/font/google"
import { useState } from "react"

const garamond = Cormorant_Garamond()

export default function Home() {
  const [showFiles, setShowFiles] = useState(false)
  const [selectedFileName, setSelectedFileName] = useState("")
  const [file, setFile] = useState<File | null>(null)
  const [classification, setClassification] = useState(null)
  const [confidence, setConfidence] = useState(null)
  const [trueLabel, setTrueLabel] = useState(null)

  const csvFiles = [
    "fraudulent_test_1.csv",
    "legitimate_test_1.csv"
  ]

  const chooseCsvFile = async (fileName: string) => {
    const response = await fetch(`/transactions/${fileName}`);

    if (!response.ok) {
      console.error("Failed to load CSV file.");
      return;
    }

    const blob = await response.blob();

    const selectedFile = new File([blob], fileName, {
      type: "text/csv"
    });

    setFile(selectedFile);
    setSelectedFileName(fileName);
    setShowFiles(false);
  }

  const display_files = (event: React.MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    setShowFiles(!showFiles);
  }
  
  const predict = async (event: React.MouseEvent<HTMLButtonElement>) => {
    // Prevents the form from being submitted and page reloading first
    event.preventDefault()

    // Make sure a file has been selected
    if (!file) {
      console.error("File has not been selected.");
      return;
    }

    // Append the file to formdata
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict",
        {
          method: "POST",
          body: formData
        }
      );
      if (!response.ok) {
        throw new Error("Failed to make prediction.");
      }

      const data = await response.json();
      setClassification(data.classification)
      setConfidence(data.confidence)
      setTrueLabel(data.true_label)
    } catch (error) {
      console.error("Failed to analyse the transaction: ", error);
    }
  }
  
  return (
    <main className="min-h-screen bg-stone-100 px-6 py-4">
      <h1 className={`${garamond.className} text-8xl font-bold text-stone-900`}>Sentry</h1>
      <div className="mx-auto max-w-3xl rounded-2xl bg-white mt-8 px-4 py-4">
        <h2 className={`${garamond.className} text-5xl font-semibold mt-2 bg-stone-100`}>
          Fraudulent Ethereum Transaction Detector
        </h2>
        <br></br>

        <form className="space-y-4">
          <button onClick={display_files} className="border border-stone-300 px-4 py-1 
          mr-4 font-medium transition hover:bg-stone-100">
            Choose transaction to make
          </button>
          <br></br>
          {
            showFiles && (
              csvFiles.map((filename) => (
                <button
                  key={filename}
                  type="button"
                  onClick={() => chooseCsvFile(filename)}
                  className="bg-stone-500 rounded-lg text-white font-medium transition
                  hover:bg-stone-700 px-4 ml-2 mr-2">
                  {filename}
                </button>
              ))
            )
          }
          <br></br>
          <button onClick={predict} className="border border-stone-300 px-4 py-1 mr-4 
          font-medium transition hover:bg-stone-100">
            Analyse transaction
          </button>
          <h3 className="mt-2 text-sm text-stone-600">
            The model will output its decision regarding the transaction.
          </h3>
          {classification && 
          <div className="mt-8 border-stone-200 bg-stone-50 font-bold">
            {`Transaction is: ${classification}`}
          </div>}
          {confidence && <h3>{`Analysed with confidence: ${confidence}%`}</h3>}
          {trueLabel && <h3>{`The transaction is actually: ${trueLabel}`}</h3>}
        </form>

        <h3 className="mt-8 text-sm text-stone-600">
          The model has not seen these transactions previously, nor will it memorise these transactions.
        </h3>
      </div>
    </main>
  )
}