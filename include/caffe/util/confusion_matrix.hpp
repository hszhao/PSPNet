/*
 * This class is modified from Stephen Gould's library: DARWIN
 * See their drwnConfusionMatrix for more reference
 *
 *
 ******************************************************************************
 ** DARWIN: A FRAMEWORK FOR MACHINE LEARNING RESEARCH AND DEVELOPMENT
 ** Distributed under the terms of the BSD license (see the LICENSE file)
 ** Copyright (c) 2007-2013, Stephen Gould
 ** All rights reserved.
 **
 ******************************************************************************
 ** FILENAME:    drwnConfusionMatrix.h
 ** AUTHOR(S):   Stephen Gould <stephen.gould@anu.edu.au>
 ** 
 ****************************************************************************
*/ 

#ifndef _CONFUSION_MATRIX_H
#define _CONFUSION_MATRIX_H

#include <vector>
#include <string>

class ConfusionMatrix {
 public:
  //create a m-by-m confusion matrix
  ConfusionMatrix();
  explicit ConfusionMatrix(const int m);
  ~ConfusionMatrix();

  void accumulate(const int actual, const int predicted);
  void accumulate(const ConfusionMatrix& conf);
 
  int numRows() const;
  int numCols() const;
  
  void resize(const int m);
  void clear();
  
  void printCounts(const char *header = NULL) const;
  void printRowNormalized(const char *header = NULL) const;
  void printColNormalized(const char *header = NULL) const;
  void printNormalized(const char *header = NULL) const;
  void printPrecisionRecall(const char *header = NULL) const;
  void printF1Score(const char *header = NULL) const;
  void printJaccard(const char *header = NULL) const;
  
  double rowSum(int n) const;
  double colSum(int m) const;
  double diagSum() const;
  double totalSum() const;
  double accuracy() const;
  double avgPrecision() const;
  double avgRecall(const bool strict = true) const;
  double avgJaccard() const;
 
  double precision(int n) const;
  double recall(int n) const;
  double jaccard(int n) const;
 
  const unsigned long& operator()(int x, int y) const;
  unsigned long& operator()(int x, int y);

 protected:
  // use unsigned long: be caureful of overflow for large-scale dataset
  std::vector< std::vector<unsigned long> > _matrix;
  
 public:
  static std::string COL_SEP;   // string for separating columns when printing
  static std::string ROW_BEGIN; // string for starting a row when printing
  static std::string ROW_END;   // string for ending a row when printing

};


#endif
