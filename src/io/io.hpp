#ifndef _IO_HPP_
#define _IO_HPP_

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <string>
#include <vector>

class CSVWriter {
  std::string fileName;
  std::string delimeter;
  int linesCount;

public:
  CSVWriter(std::string filename, std::string delm = ",")
      : fileName(filename), delimeter(delm), linesCount(0) {}
  /*
   * Member function to store a range as comma seperated value
   */
  template <typename T> void addDatainRow(T first, T last) {
    std::fstream file;
    // Open the file in truncate mode if first line else in Append Mode
    file.open(fileName,
              std::ios::out | (linesCount ? std::ios::app : std::ios::trunc));
    // Iterate over the range and add each element to file seperated by
    // delimeter.
    for (; first != last;) {
      file << *first;
      if (++first != last)
        file << delimeter;
    }
    file << "\n";
    linesCount++;
    // Close the file
    file.close();
  }
};

#endif /* end of include guard: _IO_HPP_ */
