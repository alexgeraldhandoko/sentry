import { buildModule } from "@nomicfoundation/hardhat-ignition/modules";

export default buildModule("SentryLedger", (m) => {
    // Create a Future that represents the result of deploying the
    // SentryLedger.sol smart contract
    const sentryLedger = m.contract("SentryLedger");

    // Return the Future as an export that other scripts/tasks/tests
    // can use and await to resolve
    return { sentryLedger };
})