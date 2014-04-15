

#include <iostream>
#include <fstream>

#include <cstdlib>
#include <ctime>

#define _USE_MATH_DEFINES
#include <math.h>

#include <omp.h>

#include <mkl_cblas.h>
#include <mkl_lapack.h>

#include "TDMAParabolicSolver3D.h"
#include "SolverUtil.h"


#define IDX3D(i,j,k) ((i)*ny*nz + (j)*nz + (k))

namespace {
	clock_t start_time;
	clock_t elaps_time;
}

int tdma_solve(const int n, 
	const double *b, const double *a, const double *c,
	double *x, const double *y, 
	double *gamma) 
{
	double beta = b[0];
	x[0] = y[0] / beta;

	// forward
	for (int i=1; i<n; i++) {
		gamma[i] = c[i-1] / beta;
		beta = b[i] - a[i]*gamma[i];
		x[i] = (y[i] - a[i]*x[i-1]) / beta;
	}

	// backward
	for (int i=n-2; i>=0; i--) {
		x[i] -= gamma[i+1] * x[i+1];
	}

	return 0;
} // tdma_solve


void buildTriDiag(
	const double alpha, const double beta,
	const int nx, const int ny, const int nz,
	//const int nc, 
	const double dh,
	double *b, double *a, double *c) 
{
	for (int i=0; i<nx; i++) {
		for (int j=0; j<ny; j++) {
			for (int k=0; k<nz; k++) {
				int idx = i*ny*nz + j*nz + k;

				double bb = 0, aa = 0, cc = 0;

				if (k == 0) {
					aa = 0;
					cc = -beta / (dh*dh);
					bb = alpha + beta * 2.0/(dh*dh);
				} else if (k == nz-1) {
					aa = -beta / (dh*dh);
					cc = 0;
					bb = alpha + beta * 2.0/(dh*dh);
				} else {
					aa = -beta / (dh*dh);
					cc = -beta / (dh*dh);
					bb = alpha + beta * 2.0/(dh*dh);
				}

				b[idx] = bb;
				a[idx] = aa;
				c[idx] = cc;
			}
		}
	}
} // buildTriDiag


void test_1d() {
	const double xlo = 0;
	const double xhi = 1;
	const int ncellx = 64;
	const double hx = (xhi-xlo) / ncellx;
	const int ngridx = ncellx + 1;
	const int ndofx = ngridx - 2;

	double *xpos = new double[ngridx];
	for (int i=0; i<ngridx; i++) {
		xpos[i] = xlo + hx*i;
	}


	const double alpha = 1;
	const double beta = 1;

	double *sol = new double[ngridx];
	double *rhs = new double[ngridx];
	double *ana = new double[ngridx];

	for (int i=0; i<ngridx; i++) {
		double x = xpos[i];
		ana[i] = sin(M_PI*x) + sin(4.0*M_PI*x) + sin(9.0*M_PI*x);

		rhs[i] = M_PI*M_PI * (sin(M_PI*x) + 16.0*sin(4.0*M_PI*x) + 81.0*sin(9.0*M_PI*x));
		rhs[i] += alpha * ana[i];
	}

	double *probb = new double[ndofx];
	double *proba = new double[ndofx];
	double *probc = new double[ndofx];
	buildTriDiag(alpha, beta, 1, 1, ndofx, hx, probb, proba, probc);

	double *probx = new double[ndofx];
	double *proby = new double[ndofx];
	double *probg = new double[ndofx];
	for (int i=0; i<ngridx; i++) {
		proby[i] = rhs[i+1];
	}

	tdma_solve(ndofx, probb, proba, probc, probx, proby, probg);

	for (int i=0; i<ngridx; i++) {
		sol[i] = ana[i];
		if (1<=i && i<ngridx-1) {
			sol[i] = probx[i-1];
		}
	}

	if (1) {
		std::ofstream os("hoge.csv");
		if (!os) {
			std::cerr << "Failed to open output" << std::endl;
		}

		os << "x,y,z,sol,ana" << std::endl;
		for (int i=0; i<ngridx; i++) {
			os << xpos[i] << ",0,0,"
				<< sol[i] << ","
				<< ana[i] << std::endl;
		}
	}
}

#define VTKHELPER_LINKER_PRAGMA
#include <vtkHelper.hpp>
#include <vtkRectilinearGrid.h>
#include <vtkXMLRectilinearGridWriter.h>
#include <vtkCellData.h>

void test_solver() {
	const int ndim = 3;

	const double bcpos[ndim][2] = {
		0.0, 1.0, // x low / high
		0.0, 1.0, // y low / high
		0.0, 1.0, // z low / high
	};

	//const int nnode[ndim] = { 32+2, 32+2, 32+2 };
	const int nnode[ndim] = { 64+2, 64+2, 64+2 };

	const int nx = nnode[0];
	const int ny = nnode[1];
	const int nz = nnode[2];

	const int ngrid[ndim] = { nx-2, ny-2, nz-2 };

	const int nvar = nx * ny * nz;
	const int ndof = (nx-2) * (ny-2) * (nz-2);
	std::cout << "Number of grids = " << nx << " " << ny << " " << nz << std::endl;

	const double cellsize[ndim] = {
		(bcpos[0][1] - bcpos[0][0]) / (nnode[0]-1),
		(bcpos[1][1] - bcpos[1][0]) / (nnode[1]-1),
		(bcpos[2][1] - bcpos[2][0]) / (nnode[2]-1),
	};
	const double hx = cellsize[0];
	const double hy = cellsize[1];
	const double hz = cellsize[2];

	double *nodepos[ndim];
	for (int dir=0; dir<ndim; dir++) {
		nodepos[dir] = new double[nnode[dir]];
		for (int i=0; i<nnode[dir]; i++) {
			nodepos[dir][i] = bcpos[dir][0] + cellsize[dir]*(i);
		}
	}
	const double *xpos = nodepos[0];
	const double *ypos = nodepos[1];
	const double *zpos = nodepos[2];

	//
	const double alpha = 1.0;
	const double beta = hx * 0.125;

	//
	double *fs = new double[nvar];
	double *rhs = new double[nvar];
	double *sol = new double[nvar];
	double *ana = new double[nvar];
	double *res = new double[nvar];

	// set RHS
	for (int i=0; i<nx; i++) {
		for (int j=0; j<ny; j++) {
			for (int k=0; k<nz; k++) {
				int idx = IDX3D(i,j,k);

				double x = xpos[i];
				double y = ypos[j];
				double z = zpos[k];

				double sx = sin(2*M_PI*x);
				double sy = sin(2*M_PI*y);
				double sz = sin(2*M_PI*z);
				double cx = cos(2*M_PI*x);
				double cy = cos(2*M_PI*y);
				double cz = cos(2*M_PI*z);
				
				// sin(2*pi*x) * sin(2*pi*y) * z*(1-z)
				fs[idx] = 8.0*M_PI*M_PI * sx*sy*z*(1-z) + 2.0*sx*sy;
				ana[idx] = sx * sy * z*(1-z);

				rhs[idx] = alpha*ana[idx] + beta*fs[idx];
			}
		}
	}
	// TODO take care of boundary values
	if (1) {
		//for (int i=0; i<nx; i++) {
		//	for (int j=0; j<ny; j++) {
		//		for (int k=0; k<nz; k++) {
		//			int idx = IDX3D(i,j,k);

		//			if (i == 1) {
		//				double bcval = ana[IDX3D(i-1,j,k)];
		//				fixRhs(nx, hx, rhs[idx], bctype[0][0], 0, -1, 0, bcval);
		//			} else if (i == nx-2) {
		//				double bcval = ana[IDX3D(i+1,j,k)];
		//				fixRhs(nx, hx, rhs[idx], bctype[0][1], 1, 1, 0, bcval);
		//			}
		//			if (j == 1) {
		//				double bcval = ana[IDX3D(i,j-1,k)];
		//				fixRhs(ny, hy, rhs[idx], bctype[1][0], 0, -1, 0, bcval);
		//			} else if (j == ny-2) {
		//				double bcval = ana[IDX3D(i,j+1,k)];
		//				fixRhs(ny, hy, rhs[idx], bctype[1][1], 1, 1, 0, bcval);
		//			}
		//			if (k == 1) {
		//				double bcval = ana[IDX3D(i,j,k-1)];
		//				fixRhs(nz, hz, rhs[idx], bctype[2][0], 0, -1, 0, bcval);
		//			} else if (k == nz-2) {
		//				double bcval = ana[IDX3D(i,j,k+1)];
		//				fixRhs(nz, hz, rhs[idx], bctype[2][1], 1, 1, 0, bcval);
		//			}
		//		}
		//	}
		//}
	}
	{
		elaps_time = clock() - start_time;
		std::cout << "Setup time = " << elaps_time << std::endl;
		start_time = clock();
	}

	// define a solver
	TDMAParabolicSolver3D solver(ngrid, cellsize);
	solver.setBeta(beta);

	solver.initialize();

	{
		elaps_time = clock() - start_time;
		std::cout << "Plan time = " << elaps_time << std::endl;
		start_time = clock();
	}

	// set RHS
	{
		double *solver_rhs = solver.getRhsData();
		for (int i=1; i<nx-1; i++) {
			for (int j=1; j<ny-1; j++) {
				for (int k=1; k<nz-1; k++) {
					int idx = (i-1)*(ny-2)*(nz-2) + (j-1)*(nz-2) + (k-1);
					solver_rhs[idx] = rhs[IDX3D(i,j,k)];
				}
			}
		}
	}

	// solve
	const int ncycle = 1;
	for (int cycle=0; cycle<ncycle; cycle++) {
		int ret = solver.solve();
		if (ret != 0) {
			std::cerr << "Poisson solver failed" << std::endl;
			exit(1);
		}
	}

	{
		elaps_time = clock() - start_time;
		std::cout << "Solve: Cycle=" << ncycle 
			<< "; Total time=" << elaps_time 
			<< "; Mean = " << (double) elaps_time / ncycle 
			<< std::endl;
		start_time = clock();
	}

	// retrieve solution
	{
		// fill with analytical solution
		cblas_dcopy(nvar, ana, 1, sol, 1);

		double *solver_sol = solver.getSolData();
		for (int i=1; i<nx-1; i++) {
			for (int j=1; j<ny-1; j++) {
				for (int k=1; k<nz-1; k++) {
					int idx = (i-1)*(ny-2)*(nz-2) + (j-1)*(nz-2) + (k-1);
					sol[IDX3D(i,j,k)] = solver_sol[idx];
				}
			}
		}
	}

	// residual
	for (int i=0; i<nx; i++) {
		for (int j=0; j<ny; j++) {
			for (int k=0; k<nz; k++) {
				int idx = IDX3D(i,j,k);

				res[idx] = 0;

				if (1<=i && i<nx-1 && 1<=j && j<ny-1 && 1<=k && k<nz-1) {
					double lap = 0;
					lap += 1.0/(hx*hx) * (2.0*sol[idx] - sol[IDX3D(i-1,j,k)] - sol[IDX3D(i+1,j,k)]);
					lap += 1.0/(hy*hy) * (2.0*sol[idx] - sol[IDX3D(i,j-1,k)] - sol[IDX3D(i,j+1,k)]);
					lap += 1.0/(hz*hz) * (2.0*sol[idx] - sol[IDX3D(i,j,k-1)] - sol[IDX3D(i,j,k+1)]);

					res[idx] = alpha*sol[idx] + beta*lap - rhs[idx];
				}
			}
		}
	}

	


	if (1) {
		NEW_VTKOBJ(vtkRectilinearGrid, grid);
		grid->SetDimensions(nx, ny, nz);
		// X
		NEW_VTKOBJ(vtkDoubleArray, xcoord);
		xcoord->SetName("X_COORDINATES");
		xcoord->SetNumberOfComponents(1);
		xcoord->SetNumberOfValues(nx);
		for (int i=0; i<nx; i++) 
			xcoord->SetValue(i, xpos[i]);
		grid->SetXCoordinates(xcoord);
		// Y
		NEW_VTKOBJ(vtkDoubleArray, ycoord);
		ycoord->SetName("Y_COORDINATES");
		ycoord->SetNumberOfComponents(1);
		ycoord->SetNumberOfValues(ny);
		for (int j=0; j<ny; j++) 
			ycoord->SetValue(j, ypos[j]);
		grid->SetYCoordinates(ycoord);
		// Z
		NEW_VTKOBJ(vtkDoubleArray, zcoord);
		zcoord->SetName("Z_COORDINATES");
		zcoord->SetNumberOfComponents(1);
		zcoord->SetNumberOfValues(nz);
		for (int k=0; k<nz; k++) 
			zcoord->SetValue(k, zpos[k]);
		grid->SetZCoordinates(zcoord);

		vtkSmartPointer<vtkDoubleArray> fftArray =
			vtkHelper_declareField<vtkDoubleArray>(grid, "sol", 1, nvar);
		vtkSmartPointer<vtkDoubleArray> anaArray = 
			vtkHelper_declareField<vtkDoubleArray>(grid, "ana", 1, nvar);
		vtkSmartPointer<vtkDoubleArray> resArray = 
			vtkHelper_declareField<vtkDoubleArray>(grid, "res", 1, nvar);

		for (int k=0; k<nz; k++) {
			for (int j=0; j<ny; j++) {
				for (int i=0; i<nx; i++) {
					int idx = i + j*nx + k*nx*ny;

					fftArray->SetValue(idx, sol[IDX3D(i,j,k)]);
					anaArray->SetValue(idx, ana[IDX3D(i,j,k)]);
					resArray->SetValue(idx, res[IDX3D(i,j,k)]);
				}
			}
		}

		const char filename[] = "../test/hoge00.vtr";

		grid->Update();
		NEW_VTKOBJ(vtkXMLRectilinearGridWriter, writer);
		writer->SetFileName(filename);
		writer->SetInput(grid);
		if (!writer->Write()) {
			std::cerr << "Failed to write " << filename << std::endl;
		} else {
			std::cout << "Write solution " << filename << std::endl;
		}
	}

	{
		elaps_time = clock() - start_time;
		std::cout << "IO time = " << elaps_time << std::endl;
		start_time = clock();
	}
}


int main(int argc, char *argv[]) {

	start_time = clock();
	clock_t start_time0 = start_time;

	//test_1d();
	test_solver();

	elaps_time = clock() - start_time0;
	std::cout << "Elapsed = " << elaps_time << std::endl;

	return 0;
}





