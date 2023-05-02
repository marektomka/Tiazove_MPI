#define _USE_MATH_DEFINES
#include <iostream>
#include <mpi.h>
#include <time.h>  
#include <fstream>
#include <iostream>
#include <string>
#include <math.h>

using namespace std;

//#define N 902     // počet uzlov
//#define D 500
//#define R 6378    // polomer Zeme 

#define N 8102
#define D 200000
#define R 6378000
     
// #define N 160002
// #define D 50000
// #define R 6378000

#define GM 398600.5   // geocentrická gravitačná konštanta
#define maxIter 100   // max. pocet iteracii
#define tol 0.0000001  // tolerancia 



int main(int argc, char** argv) {

    // inicializacia paralelneho programu // 
    int nprocs, myrank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    int nlocal = N / nprocs + 1;
    int istart = myrank * nlocal;
    
    // alokacia vektorov
    double* B = new double[N];  //zemepisná šírka
    double* L = new double[N];  //zemepisná dĺžka
    double* H = new double[N];  //výška
    double* g = new double[N];  //g
    double* un = new double[N]; //druhá derivácia u podľa normály
    double* XX = new double[N]; 
    double* XY = new double[N];
    double* XZ = new double[N]; // súradnice bodov na Zemi XX, XY, XZ
    double* SX = new double[N];
    double* SY = new double[N];
    double* SZ = new double[N];  // súradnice bodov na fiktívnej hranici SX, SY, SZ

    double* nx = new double[N];
    double* ny = new double[N];
    double* nz = new double[N];  // súradnice norm. vektora

    // matica a pre MPI
    double* aa = new double[nlocal * N];

    // pomocne premenne 
    int iter = 0;
    double normS = 0;
    double rhoPrev = 0.0, rhoAct = 0.0;
    double ts = 0.0, tt = 0.0;
    double rv = 0.0;
    double normStemp = 0.0;
    double beta = 0.0, alfa = 0.0, omega = 0.0;
    double normR = 0.0;
    double resid = 0;


    // FILE READ by 0. process // 
    if (myrank == 0) 
    {
        std::ifstream myfile;
        std::string mystring;
        //myfile.open("BL-902.dat");
        myfile.open("BL-8102.dat");
        // myfile.open("BL-160002.dat");

        if (myfile.is_open())
        {
            for (int i = 0; i < N; i++)
            {
                myfile >> B[i];
                myfile >> L[i];
                myfile >> H[i];
                myfile >> g[i];   // right side
                myfile >> un[i];
            }
        }
        myfile.close();

        // SET COORDINATES  Xi, Si // 
        for (int i = 0; i < N; i++)
        {
            // Suradnice zeme
            //XX[i] = R * std::cos(B[i] * (M_PI / 180.0)) * std::cos(L[i] * (M_PI / 180.0));    
            XX[i] = (R + H[i]) * std::cos(B[i] * (M_PI / 180.0)) * std::cos(L[i] * (M_PI / 180.0));
            //XY[i] = R * std::cos(B[i] * (M_PI / 180.0)) * std::sin(L[i] * (M_PI / 180.0));
            XY[i] = (R + H[i]) * std::cos(B[i] * (M_PI / 180.0)) * std::sin(L[i] * (M_PI / 180.0));
            //XZ[i] = R * std::sin(B[i] * (M_PI / 180.0));
            XZ[i] = (R + H[i]) * std::sin(B[i] * (M_PI / 180.0));

            // Suradnice na fiktivnej hranici
            //SX[i] = (R - D) * std::cos(B[i] * (M_PI / 180.0)) * std::cos(L[i] * (M_PI / 180.0));    
            SX[i] = (R + H[i] - D) * std::cos(B[i] * (M_PI / 180.0)) * std::cos(L[i] * (M_PI / 180.0));
            //SY[i] = (R - D) * std::cos(B[i] * (M_PI / 180.0)) * std::sin(L[i] * (M_PI / 180.0));
            SY[i] = (R + H[i] - D) * std::cos(B[i] * (M_PI / 180.0)) * std::sin(L[i] * (M_PI / 180.0));
            //SZ[i] = (R - D) * std::sin(B[i] * (M_PI / 180.0));
            SZ[i] = (R + H[i] - D) * std::sin(B[i] * (M_PI / 180.0));

            // Norm vector
            nx[i] = -XX[i] / sqrt(XX[i] * XX[i] + XY[i] * XY[i] + XZ[i] * XZ[i]);
            ny[i] = -XY[i] / sqrt(XX[i] * XX[i] + XY[i] * XY[i] + XZ[i] * XZ[i]);
            nz[i] = -XZ[i] / sqrt(XX[i] * XX[i] + XY[i] * XY[i] + XZ[i] * XZ[i]);

        }

    }

    MPI_Bcast(B, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(L, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(H, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(g, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(un, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(XX, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(XY, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(XZ, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(SX, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(SY, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(SZ, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(nx, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(ny, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(nz, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    


    // COMPUTE Aij // 
    for (int i = 0; i < nlocal; i++){
        for (int j = 0; j < N; j++)
        {
            int iGlobal = i + myrank * nlocal;
            double rx = XX[iGlobal] - SX[j];
            double ry = XY[iGlobal] - SY[j];
            double rz = XZ[iGlobal] - SZ[j];

            double k = rx * nx[iGlobal] + ry * ny[iGlobal] + rz * nz[iGlobal];
            double r = sqrt(rx * rx + ry * ry + rz * rz);

            aa[i * N + j] = k / (4 * M_PI * r * r * r);

        }
        
    }

   
    // BCG ALGORITHM // 

    // zaciatocna podmienka
    double g0 = -GM / (R * R);

    // pomocne vektory
    double* s = new double[N];
    double* p = new double[N];
    double* t = new double[N + 100] {0.0};
    double* v = new double[N + 100] {0.0};

    double* x = new double[N] {0.0};
    double* r = new double[N];   //vektor r
    double* r0 = new double[N];  //rez

    // pomocne vektory MPI
    double* tempv = new double[nlocal] {0.0};
    double* tempt = new double[nlocal] {0.0};

    

    // Začiatočné podmienky
    for (int i = 0; i < N; i++) {
        x[i] = 0;
        //r[i] = g0;
        //r0[i] = g0;
        r[i] = -g[i] * 0.00001;
        r0[i] = -g[i] * 0.00001;
    }
    double norm = 0;
    for (int j = 0; j < N; j++) {
        norm = norm + r0[j] * r0[j];
    }

    if (myrank == 0) {
        std::cout << "Norm of residuals: " << sqrt(norm) << "\n";
    }
        

    // BCG algortihm // 
    for (int iter = 1; iter < maxIter; iter++)
    {  
        // rhoPrev, rhoAct
        rhoPrev = rhoAct;
        rhoAct = 0.0;
        for (int i = 0; i < N; i++)
        {
            rhoAct += r0[i] * r[i];
        }

        // Kontrola rhoAct
        if (rhoAct == 0.00) {
            printf("Method fails!!!\n");
            break;
        }

        // vektor p - prva iteracia
        if (iter == 1)
        {
            for (int i = 0; i < N; i++)
            { 
                p[i] = r0[i]; 
            }
        }
        // vektor p - ostatne iteracie
        else 
        {   
            beta = (rhoAct / rhoPrev) * (alfa / omega);
            
            for (int i = 0; i < N; i++)
            {
                p[i] = r0[i] + beta * (p[i] - omega * v[i]);
            }
        }

        // vektor v
        for (int i = 0; i < nlocal; i++)
        {
            tempv[i] = 0.0;
            for (int j = 0; j < N; j++)
            {
                tempv[i] += aa[i * N + j] * p[j];
            }
        }

        MPI_Allgather(tempv, nlocal, MPI_DOUBLE, v, nlocal, MPI_DOUBLE, MPI_COMM_WORLD);

        // alfa
        rv = 0.0;
        for (int i = 0; i < N; i++)
        {
            rv += r[i] * v[i];
        }
        alfa = rhoAct / rv;

        // vektor s, norm of rez
        normStemp = 0.0;
        for (int i = 0; i < N; i++)
        {
            s[i] = r0[i] - alfa * v[i];
            normStemp += s[i] * s[i];
        }
        normS = sqrt(normStemp);

        // kontrola norm of rez, vektor x
        if (normS < tol)
        {
            for (int i = 0; i < N; i++)
            {
                x[i] += alfa * p[i];
            }
            break;
        }
   
        // vektor t
        for (int i = 0; i < nlocal; i++)
        {
          tempt[i] = 0.0;
          for (int j = 0; j < N; j++)
            {
                tempt[i] += aa[i * N + j] * s[j];
            }
        }

        MPI_Allgather(tempt, nlocal, MPI_DOUBLE, t, nlocal, MPI_DOUBLE, MPI_COMM_WORLD);

        // omega
        ts = 0.0;
        tt = 0.0;
        for (int i = 0; i < N; i++)
        {
            ts += t[i] * s[i];
            tt += t[i] * t[i];
        }
        omega = ts / tt;

        // vektor rez, norm of rez
        normR = 0.0;
        for (int i = 0; i < N; i++)
        {
            x[i] += alfa * p[i] + omega * s[i];
            r0[i] = s[i] - omega * t[i];
            normR += r0[i] * r0[i];
        }
        resid = sqrt(normR);

        if (myrank == 0) {
            printf("Norm of residuals: %.20lf\n", resid);
        }
            
        // ukoncenie cyklu
        if (resid < tol || omega == 0) {
            break;
        }

    }
   
    //deallokovanie vektorov
    delete[] aa; delete[] r0; delete[] r; 
    delete[] v; delete[] p; delete[] s; delete[] t;

    //vektor u - reálne dáta
    double* u = new double[N + 100] {0};
    double* tempu = new double[nlocal];
    for (int i = 0; i < nlocal; i++) {
        double total = 0;
        int iGlobal = i + myrank * nlocal;
        for (int j = 0; j < N; j++) {
            total += 1 / (4 * M_PI * (sqrt(pow(XX[iGlobal] - SX[j], 2) + pow(XY[iGlobal] - SY[j], 2) + pow(XZ[iGlobal] - SZ[j], 2)))) * x[j];
        }
        tempu[i] = total;
    }
    
    MPI_Allgather(tempu, nlocal, MPI_DOUBLE, u, nlocal, MPI_DOUBLE, MPI_COMM_WORLD);
    
    // FILE WRITE // 
    if (myrank == 0)
    {
        fstream myfileW;

        //myfileW.open("Output902.dat", std::ios::out);
        myfileW.open("Output8102.dat", std::ios::out);
        //myfileW.open("Output160002.dat", std::ios::out);
        
        if (!myfileW)
        {
            std::cout << "File not created\n";
        }
        else
        {
            for (int i = 0; i < N; i++)
            {
                myfileW << B[i] << " " << L[i] << " " << u[i] << "\n";
            }
            myfileW.close();

            std::cout << "File created succesful\n";

        }
    }

    //deallokovanie vektorov
    delete[] B; delete[] L; delete[] H; delete[] g; delete[] un; delete[] u; delete[] x;
    delete[] tempu; delete[] tempt; delete[] tempv;
    delete[] SX; delete[] SY; delete[] SZ;
    delete[] XX; delete[] XY; delete[] XZ;
    delete[] nx; delete[] ny; delete[] nz;
    

    MPI_Finalize();
}