#include <tr1/cmath>
#include <iostream>
#include <string>
#include <fstream> 

int main() {
    for (int i = 0; i <= 100000; ++i) {
        double x = ((1.0e3-1.0e-8)/100000) * i + 1.0e-8;
        double G = std::tr1::cyl_bessel_k(2.0/3.0 , x);
    }
  
  //get array size
  const int numPoints = 1000;
  double minValue = 1.0e-5;  // Start point (minimum value)
  double maxValue = 1.0e6; // End point (maximum value)

  //exception handling
  try {

    std::cout << "\nWriting  array contents to file...";

    //open file for writing
    std::ofstream fw("../../python/G.txt", std::ofstream::out);

    //check if file was successfully opened for writing
    if (fw.is_open())
    {
      //store array contents to text file

     for (int i = 0; i < numPoints; ++i) {
        double x = minValue * std::pow(maxValue / minValue, i / static_cast<double>(numPoints - 1));
         fw << x * std::tr1::cyl_bessel_k(2.0/3.0 , x)  << "\n";
     }
      fw.close();
    }
    else std::cout << "Problem with opening file";

  }
  catch (const char* msg) {
    std::cerr << msg << std::endl;
  }

  std::cout << "\nDone!";
  std::cout << "\nPress any key to exit...";
  getchar();

}