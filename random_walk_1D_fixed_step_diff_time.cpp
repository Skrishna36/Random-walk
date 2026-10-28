// ==========================================================
// 1D Random Walk with Fixed Step Size at Different Times
// ----------------------------------------------------------
// This program simulates many particles performing 1D random
// walks on a lattice. Each particle starts at the origin and
// takes "N" steps of fixed size 1, moving +1 or -1 with equal
// probability at every step.
//
// The simulation is repeated for three walk lengths (N = 50,
// 100, 200). For each N, the final positions of all particles
// are recorded in a histogram. The distribution of final
// positions approaches a Gaussian with variance N, due to the
// central limit theorem. Scaling the position by sqrt(N) and
// the probability by sqrt(N) collapses all three curves onto
// a single Gaussian.
//
// Output: "walk4.dat" containing, for each N, a block of
// (x, P(x)*sqrt(N), x/sqrt(N), theoretical Gaussian),
// with blocks separated by two blank lines.
// ==========================================================
//g++ -O3 -march=native random_walk_1D_fixed_step_diff_time.cpp -o random_walk_1D_fixed_step_diff_time
//./random_walk_1D_fixed_step_diff_time
// ==========================================================
#include <bitset>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

std::vector<long long> endpoint_histogram(int N, long long M, std::mt19937_64& gen) {
    std::vector<long long> hist(2 * N + 1, 0);
    for (long long p = 0; p < M; ++p) {
        int up = 0;        
        int left = N;
        while (left >= 64) {                    
            up += (int)std::bitset<64>(gen()).count();
            left -= 64;
        }
        if (left > 0)
            up += (int)std::bitset<64>(gen() & ((1ULL << left) - 1)).count();
        int pos = 2 * up - N;                   
        ++hist[pos + N];
    }
    return hist;
}

int main(int argc, char* argv[]) {
    const long long M = (argc > 1) ? std::atoll(argv[1]) : 100000;
    const int steps[] = {50, 100, 200};
    const double PI = std::acos(-1.0);

    std::random_device rd;
    std::seed_seq ss{rd(), rd(), rd(), rd()};
    std::mt19937_64 gen(ss);                   

    std::ofstream fout("walk4.dat");
    std::cout << "Final position vs number of particles:\n";

    for (int N : steps) {
        std::vector<long long> hist = endpoint_histogram(N, M, gen);
        const double sN = std::sqrt((double)N);

        auto header = [&](std::ostream& os) {
            os << "# N = " << N << "\n# x  P(x)*sqrt(N)  x/sqrt(N)  theory\n";
        };
        header(std::cout);
        header(fout);

        for (int x = -N; x <= N; x += 2) {      
            double scaled = (double)hist[x + N] / M * sN;
     
            double theory = 2.0 / std::sqrt(2.0 * PI) * std::exp(-(double)x * x / (2.0 * N));
            auto row = [&](std::ostream& os) {
                os << x << ' ' << scaled << ' ' << x / sN << ' ' << theory << '\n';
            };
            row(std::cout);
            row(fout);
        }
        std::cout << '\n';
        fout << "\n\n";                       
    }
    return 0;
}
