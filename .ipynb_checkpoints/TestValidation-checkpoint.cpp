#include "SharedProtocol.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

int main() {
  std::ifstream df("dataset.bin", std::ios::binary),
      tf("msft_truth_mbp-10.bin", std::ios::binary);
  std::vector<LOBSnapshot> u;
  std::vector<TruthMessage> t;
  LOBSnapshot us;
  TruthMessage ts;
  while (df.read((char *)&us, sizeof(LOBSnapshot)))
    u.push_back(us);
  while (tf.read((char *)&ts, sizeof(TruthMessage)))
    t.push_back(ts);

  long matches = 0, checked = 0;
  size_t uIdx = 0;
  for (const auto &truth : t) {
    while (uIdx < u.size() - 1 && u[uIdx + 1].time <= truth.time)
      uIdx++;
    if (std::abs(u[uIdx].time - truth.time) < 0.001) {
      if (u[uIdx].bidPrice[0] == truth.bidPrice &&
          u[uIdx].askPrice[0] == truth.askPrice)
        matches++;
      checked++;
    }
  }
  std::cout << "Precision Accuracy: "
            << (checked > 0 ? (double)matches / checked * 100 : 0) << "%"
            << std::endl;
  return 0;
}