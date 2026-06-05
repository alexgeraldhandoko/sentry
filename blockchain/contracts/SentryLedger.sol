// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

contract SentryLedger {
    // Declare the transaction record format
    struct TransactionRecord {
        string classification;
        uint256 confidence;
        string trueLabel;
        uint256 timestamp;
    }

    TransactionRecord[] private transactionHistory;

    function recordTransaction(
        string memory classification,
        uint256 confidence,
        string memory trueLabel
    ) public {
        // Create new transaction record object
        TransactionRecord memory newTransaction = TransactionRecord(
            classification, confidence, trueLabel, block.timestamp);

        // Push the new transaction into the history
        transactionHistory.push(newTransaction);
    }

    function getTransactionCount() public view returns (uint256) {
        return transactionHistory.length;
    }

    function getTransactionAtIndex(
        uint256 index
    ) public view returns (TransactionRecord memory) {
        return transactionHistory[index];
    }
}