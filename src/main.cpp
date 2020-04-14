#include "framework/environment.hpp"
#include <iostream>

using namespace std;
using namespace Aperture;

int main(int argc, char *argv[]) {
  sim_environment env(&argc, &argv);
  env.init();
  env.run();
  return 0;
}
