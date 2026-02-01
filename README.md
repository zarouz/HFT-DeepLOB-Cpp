# Market Data Pipeline

This project consists of four C++20 components that together simulate a simple market data flow with integrity checking.

## Components

* **IntegrityChecker (`checker`)**
  Reads `msft_truth.bin` automatically and validates the integrity of the data stream.

* **DataRecorder (`recorder`)**
  Writes recorded output to `dataset.bin` automatically.

* **MarketDataReceiver (`engine`)**
  Core engine that receives and processes market data. Runs with multithreading enabled.

* **MarketDataSender (`exchange`)**
  Reads `msft_orders.bin` automatically and sends market data into the system.

---

## Build Instructions

Use the following exact commands to compile each component (C++20 required):

```bash
g++ -std=c++20 DataRecorder.cpp -o recorder
g++ -std=c++20 IntegrityChecker.cpp -o checker
g++ -std=c++20 MarketDataReceiver.cpp -o engine -pthread
g++ -std=c++20 MarketDataSender.cpp -o exchange
```

---

## Run Instructions (Important)

The programs **must be executed in the following order**, each in its own terminal:

1. **Terminal 1** – Integrity Checker

   ```bash
   ./checker
   ```

   Automatically reads `msft_truth.bin`.

2. **Terminal 2** – Data Recorder

   ```bash
   ./recorder
   ```

   Automatically writes `dataset.bin`.

3. **Terminal 3** – Market Data Engine

   ```bash
   ./engine
   ```

4. **Terminal 4** – Market Data Exchange

   ```bash
   ./exchange
   ```

   Automatically reads `msft_orders.bin`.

⚠️ **Running the programs out of order may lead to incorrect behavior or failures.**

---

## Notes

* Ensure all required `.bin` files are present in the working directory before execution.
* Tested with `g++` supporting the C++20 standard.
