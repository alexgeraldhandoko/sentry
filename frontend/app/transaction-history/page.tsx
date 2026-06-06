"use client"
import { Cormorant_Garamond } from "next/font/google";
import { useState, useEffect } from "react";
import TransactionRecordBox from "./TransactionRecordBox";

const garamond = Cormorant_Garamond({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

type TransactionRecord = {
    classification: string,
    confidence: number,
    trueLabel: string,
    timestamp: number
}

export default function TransactionHistory() {
    const [transactionHistory, setTransactionHistory] = useState<TransactionRecord[] | null>(null);

    const getTransactionHistory = async () => {
        const response = await fetch("http://127.0.0.1:8000/get-transaction-history");

        if (!response.ok) {
            console.error("Failed to get transaction history.");
            return;
        }

        // Data is an array of transaction objects
        const data = await response.json();

        console.log("Transaction history from backend: ", data);

        setTransactionHistory(data);
    }

    useEffect(() => {
        getTransactionHistory()
    }, [])

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
              <a href="/" className="transition hover:text-[#6b1f2a]">
                Detector
              </a>

              <a
                href="/transaction-history"
                className="text-[#6b1f2a] transition hover:text-[#6b1f2a]"
              >
                Transaction History
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
            Transaction History
          </h1>

          <p
            className={`${garamond.className} mt-2 text-xl text-[#6d5a43] md:text-2xl`}
          >
            Immutable Timeline of Analysed Ethereum Transactions
          </p>
        </header>

        {/* Main history section */}
        <section className="mt-8 grid border border-[#c8b18a] bg-[#fffaf2] lg:grid-cols-[0.78fr_1.22fr]">
          {/* Summary panel */}
          <aside className="border-b border-[#c8b18a] bg-[#fbf4e9] p-6 lg:border-b-0 lg:border-r">
            <div className="border-b border-[#c8b18a] pb-5">
              <p className="text-[11px] font-semibold uppercase tracking-[0.28em] text-[#8a6b3d]">
                Ledger Overview
              </p>

              <h2
                className={`${garamond.className} mt-3 text-3xl font-semibold text-[#2b211b] md:text-4xl`}
              >
                Recorded Decisions
              </h2>

              <p className="mt-4 text-sm leading-6 text-[#67594d]">
                Each transaction shown here represents a model decision that has
                been recorded into the transaction ledger.
              </p>
            </div>

            <div className="mt-6 divide-y divide-[#d8c6aa] border border-[#c8b18a] bg-[#fffaf2]">
              <div className="p-4">
                <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-[#8a6b3d]">
                  Total Records
                </p>

                <p
                  className={`${garamond.className} mt-2 text-3xl font-semibold text-[#2b211b]`}
                >
                    {transactionHistory?.length}
                </p>
              </div>

              <div className="p-4">
                <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-[#8a6b3d]">
                  Fraudulent
                </p>

                <p
                  className={`${garamond.className} mt-2 text-3xl font-semibold text-[#6b1f2a]`}
                >
                  {transactionHistory?.filter((transaction) => 
                    transaction.classification == "Fraudulent").length}
                </p>
              </div>

              <div className="p-4">
                <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-[#8a6b3d]">
                  Legitimate
                </p>

                <p
                  className={`${garamond.className} mt-2 text-3xl font-semibold text-[#3f4f2e]`}
                >
                  {transactionHistory?.filter((transaction) => 
                    transaction.classification == "Legitimate").length}
                </p>
              </div>
            </div>
          </aside>

          {/* Timeline */}
          <section className="p-6">
            <div className="border-b border-[#d8c6aa] pb-5">
              <p className="text-[11px] font-semibold uppercase tracking-[0.28em] text-[#8a6b3d]">
                Transaction Timeline
              </p>

              <h2
                className={`${garamond.className} mt-3 text-3xl font-semibold text-[#2b211b] md:text-4xl`}
              >
                Latest Transactions
              </h2>

              <p className="mt-4 max-w-2xl text-sm leading-6 text-[#67594d]">
                Chronological record of analysed
                transactions, including model classification, confidence, true
                label, timestamp, and ledger hash details.
              </p>
            </div>
            {transactionHistory && 
                transactionHistory.map((transaction) => (
                    <TransactionRecordBox
                        key={transaction.timestamp}
                        classification={transaction.classification}
                        confidence={transaction.confidence}
                        trueLabel={transaction.trueLabel}
                        timestamp={transaction.timestamp}
                    />
                ))
            }
          </section>
        </section>
      </div>
    </main>
  );
}