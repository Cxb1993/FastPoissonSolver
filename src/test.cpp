
#include <iostream>
#include <fstream>

#include <cstdlib>
#include <ctime>

#define _USE_MATH_DEFINES
#include <math.h>

#include <omp.h>

#include <mkl_cblas.h>

#include <fftw3.h>
#pragma comment(lib, "libfftw3-3.lib")
//#pragma comment(lib, "libfftw-3.3.lib")


#include "FastPoisson.h"

#define IDX2D(i,j) ((i)*ny + (j))
#define IDX3D(i,j,k) ((i)*ny*nz + (j)*nz + (k))

namespace {
	clock_t start_time;
	clock_t elaps_time;
}

void fixRhs(int n, double h, double &rhs, 
	int bctype, int bcside, double bcdir, double bcpos, double bcval) 
{
	switch (bctype) {
	case BCType_DIR:
		rhs += 2.0 * bcval / (h*h);
		break;
	case BCType_NEU:
		rhs += bcdir * bcval / h;
		break;
	case BCType_CDIR:
		rhs += bcval / (h*h);
		break;
	default:
		break;
	}
}

void selectTrans(bool isfwd,
	int n, double h, int bc_lo, int bc_hi,
	fftw_r2r_kind &fftwkind, double &normalizer) 
{
	if (bc_lo==BCType_PER && bc_hi==BCType_PER) {
		if (isfwd) {
			fftwkind = FFTW_R2HC;
			normalizer = 1.0 / n;
		} else {
			fftwkind = FFTW_HC2R;
			//normalizer = 0.5;
			normalizer = 1.0;
		}
	} else if (bc_lo==BCType_DIR && bc_hi==BCType_DIR) {
		if (isfwd) {
			fftwkind = FFTW_RODFT10;
			normalizer = 1.0 / n;
		} else {
			fftwkind = FFTW_RODFT01;
			normalizer = 0.5;
		}
	} else if (bc_lo==BCType_DIR && bc_hi==BCType_NEU) {
		if (isfwd) {
			fftwkind = FFTW_RODFT11;
			normalizer = 1.0 / n;
		} else {
			fftwkind = FFTW_RODFT11;
			normalizer = 0.5;
		}
	} else if (bc_lo==BCType_NEU && bc_hi==BCType_DIR) {
		if (isfwd) {
			fftwkind = FFTW_REDFT11;
			normalizer = 1.0 / n;
		} else {
			fftwkind = FFTW_REDFT11;
			normalizer = 0.5;
		}
	} else if (bc_lo==BCType_NEU && bc_hi==BCType_NEU) {
		if (isfwd) {
			fftwkind = FFTW_REDFT10;
			normalizer = 1.0 / n;
		} else {
			fftwkind = FFTW_REDFT01;
			normalizer = 0.5;
		}
	}
	else {
		std::cerr << __FUNCTION__ << ": Invalid BC pairing: "
			<< "LO=" << bc_lo << ", HI="<< bc_hi
			<< std::endl;
		exit(1);
	}
}

void computeLambda(int n, double h, int bc_lo, int bc_hi, double *lambda) {
	const double h2inv = 1.0 / (h*h);

	if (bc_lo==BCType_PER && bc_hi==BCType_PER) {
		const double w = M_PI / n;
		lambda[0] = 0;
		lambda[n-1] = 4;
		for (int i=1; i<n/2; i++) {
			lambda[i*2-1] = 4.0 * pow(sin(w*i),2) * h2inv;
			lambda[i*2] = lambda[i*2-1];
		}
	} else if (bc_lo==BCType_DIR && bc_hi==BCType_DIR) {
		const double w = M_PI / n;
		for (int i=0; i<n; i++) {
			lambda[i] = (2.0 - 2.0*cos(w*(i+1))) * h2inv;
		}
	} else if (bc_lo==BCType_DIR && bc_hi==BCType_NEU) {
		const double w = M_PI / (2*n);
		for (int i=0; i<n; i++) {
			lambda[i] = (2.0 - 2.0*cos(w*(2*i+1))) * h2inv;
		}
	} else if (bc_lo==BCType_NEU && bc_hi==BCType_DIR) {
		const double w = M_PI / (2*n);
		for (int i=0; i<n; i++) {
			lambda[i] = (2.0 - 2.0*cos(w*(2*i+1))) * h2inv;
		}
	} else if (bc_lo==BCType_NEU && bc_hi==BCType_NEU) {
		const double w = M_PI / n;
		// first eigenvalue is zero
		//lambda[0] = DBL_MAX;
		lambda[0] = 0;
		for (int i=1; i<n; i++) {
			lambda[i] = (2.0 - 2.0*cos(w*i)) * h2inv;
		}
	}
	else {
		std::cerr << __FUNCTION__ << ": Invalid BC pairing: "
			<< "LO=" << bc_lo << ", HI="<< bc_hi
			<< std::endl;
		exit(1);
	}
}

inline double computeLambda(int n, int i, int bc_lo, int bc_hi) {
	double lambda = 0;
	if (bc_lo==BCType_DIR && bc_hi==BCType_DIR) {
		const double w = M_PI / n;
		lambda = (2.0 - 2.0*cos(w*(i+1)));
	} else if (bc_lo==BCType_DIR && bc_hi==BCType_NEU) {
		const double w = M_PI / (2*n);
		lambda = (2.0 - 2.0*cos(w*(2*i+1)));
	} else if (bc_lo==BCType_NEU && bc_hi==BCType_DIR) {
		const double w = M_PI / (2*n);
		lambda = (2.0 - 2.0*cos(w*(2*i+1)));
	} else if (bc_lo==BCType_NEU && bc_hi==BCType_NEU) {
		const double w = M_PI / n;
		// NOTE first eigenvalue is zero
		lambda = (2.0 - 2.0*cos(w*i));
	}
	else {
		std::cerr << __FUNCTION__ << ": Invalid BC pairing: "
			<< "LO=" << bc_lo << ", HI="<< bc_hi
			<< std::endl;
		exit(1);
	}
	return lambda;
}

void computeLambda(double *lambda,
	int nx, double hx, const int bctypex[2], 
	int ny, double hy, const int bctypey[2]) 
{
	const double hx2inv = 1.0 / (hx*hx);
	const double hy2inv = 1.0 / (hy*hy);

	for (int i=0; i<nx; i++) {
		for (int j=0; j<ny; j++) {
			int idx = IDX2D(i,j);
			lambda[idx] = 0;

			double eigx = computeLambda(nx, i, bctypex[0], bctypex[1]);
			double eigy = computeLambda(ny, j, bctypey[0], bctypey[1]);

			lambda[idx] = eigx*hx2inv + eigy*hy2inv;
		}
	}
}


inline double func_phi(double x) {
	double pix = M_PI * x;
	double phi = sin(pix) + sin(4.0*pix) + sin(9.0*pix);
	return phi;
}
inline double func_dphi(double x) {
	double pix = M_PI * x;
	double dphi = M_PI * (cos(pix) + 4.0*cos(4.0*pix) + 9.0*cos(9.0*pix));
	return dphi;
}
inline double func_d2phi(double x) {
	double pix = M_PI * x;
	double d2phi = -M_PI*M_PI * (sin(pix) + 16.0*sin(4.0*pix) + 81.0*sin(9.0*pix));
	return d2phi;
}

int main1d(int argc, char *argv[]) {
	
	const double xlo = 0.0;
	const double xhi = 1.0;
	//const int nx = 64;
	//const int nx = 128;
	//const int nx = 256;
	const int nx = 512;

	const int bctype[2] = { 
		//BCType_DIR, BCType_DIR,
		//BCType_DIR, BCType_NEU,
		//BCType_NEU, BCType_DIR,
		BCType_NEU, BCType_NEU,
	};
	double bcval[2] = {0, };
	if (bctype[0] == BCType_DIR) {
		bcval[0] = func_phi(xlo);
	} else if (bctype[0] == BCType_NEU) {
		bcval[0] = func_dphi(xlo);
	}
	if (bctype[1] == BCType_DIR) {
		bcval[1] = func_phi(xhi);
	} else if (bctype[1] == BCType_NEU) {
		bcval[1] = func_dphi(xhi);
	}

	const double hx = (xhi-xlo) / nx;
	const double hx2 = hx * hx;

	//
	//const double alpha = 1.0;
	const double alpha = 0.0;

	//
	double *xs = NULL;
	double *fs = NULL;
	double *rhs = NULL;
	double *fhat = NULL;
	double *lambda = NULL;
	double *sol = NULL;

	xs = (double*) fftw_malloc(sizeof(*xs) * nx);
	fs = (double*) fftw_malloc(sizeof(*fs) * nx);
	rhs = (double*) fftw_malloc(sizeof(*rhs) * nx);
	fhat = (double*) fftw_malloc(sizeof(*fhat) * nx);
	lambda = (double*) fftw_malloc(sizeof(*lambda) * nx);
	sol = (double*) fftw_malloc(sizeof(*sol) * nx);

	// boundary condition
	for(int i=0; i<nx; i++) {
		xs[i] = xlo + hx * (i+0.5);

		double phi = func_phi(xs[i]);
		double d2phi = func_d2phi(xs[i]);

		fs[i] = -d2phi + alpha*phi;
		rhs[i] = fs[i];

		if (i == 0) {
			fixRhs(nx, hx, rhs[i], bctype[0], 0, -1.0, 0.0, bcval[0]);
			std::cout << "BC_LO=" << bctype[0] << std::endl;
		} else if (i == nx-1) {
			fixRhs(nx, hx, rhs[i], bctype[1], 1, 1.0, 0.0, bcval[1]);
			std::cout << "BC_HI=" << bctype[1] << std::endl;
		}
	}

	// compute perturbation
	double perturb = 0;
	if (alpha == 0) {
		if (bctype[0]==BCType_NEU && bctype[1]==BCType_NEU) {
			perturb = 0;
			for (int i=0; i<nx; i++) {
				perturb += rhs[i];
			}
			perturb /= nx;
			for (int i=0; i<nx; i++) {
				rhs[i] -= perturb;
			}
		}
	}
	std::cout << "PERTURB=" << perturb << std::endl;


	{
		fftw_r2r_kind fftwkind;
		double normalizer;
		selectTrans(true, nx, hx, bctype[0], bctype[1], fftwkind, normalizer);
		//
		//fftw_plan plan = fftw_plan_r2r_1d(nx, rhs, fhat, FFTW_RODFT10, FFTW_ESTIMATE);
		fftw_plan plan = fftw_plan_r2r_1d(nx, rhs, fhat, fftwkind, FFTW_ESTIMATE);
		//
		fftw_execute(plan);
		//
		fftw_destroy_plan(plan);

		// regularize
		for(int i=0; i<nx; i++) {
			//fhat[i] /= nx;
			fhat[i] *= normalizer;
		}
	}

	
#if (0)
	if (0) {
		fftw_plan plan = fftw_plan_r2r_1d(nx, fhat, sol, FFTW_RODFT01, FFTW_ESTIMATE);
		fftw_execute(plan);
		fftw_destroy_plan(plan);
		// regularize
		for(int i=0; i<nx; i++) {
			sol[i] *= 0.5;
		}


		for(int i=0; i<nx; i++) {
			std::cout << i << "," << sol[i] / rhs[i] << std::endl;
		}
	}
#endif

	{
		//const double omega = M_PI / (2*nx);
		//for(int i=0; i<nx; i++) {
		//	//double lambda = 4.0 * pow(sin(omega*(i+1)), 2);
		//	double lambda = 2.0 - 2.0*cos(M_PI*(i+1)/nx);
		//	fhat[i] = fhat[i] / (lambda/hx2 + alpha);
		//}

		//
		computeLambda(nx, hx, bctype[0], bctype[1], lambda);
		//
		for (int i=0; i<nx; i++) {
			if (lambda[i]==0 && alpha==0) {
				fhat[i] = 0;
			} else {
				fhat[i] = fhat[i] / (lambda[i] + alpha);
			}
		}


		// transform back to physical space
		fftw_r2r_kind fftwkind;
		double normalizer;
		selectTrans(false, nx, hx, bctype[0], bctype[1], fftwkind, normalizer);

		//fftw_plan plan = fftw_plan_r2r_1d(nx, fhat, sol, FFTW_RODFT01, FFTW_ESTIMATE);
		fftw_plan plan = fftw_plan_r2r_1d(nx, fhat, sol, fftwkind, FFTW_ESTIMATE);
		fftw_execute(plan);
		fftw_destroy_plan(plan);
		// regularize
		for(int i=0; i<nx; i++) {
			//sol[i] *= 0.5;
			sol[i] *= normalizer;
		}

		//for(int i=0; i<nx; i++) {
		//	std::cout << i << "," << sol[i]/func_phi(xs[i]) << std::endl;
		//}

	}

	{
		std::ofstream ofs("../test/hoge.csv");
		ofs << "x,ana,fft,err,res" << std::endl;
		for(int i=0; i<nx; i++) {
			double ana = func_phi(xs[i]);
			double err = sol[i]/ana - 1;

			// compute residual
			double res = 0;
			if (i == 0) {
				double solext = 0;
				if (bctype[0] == BCType_DIR) {
					solext = 2.0*bcval[0] - sol[i];
				} else if (bctype[0] == BCType_NEU) {
					solext = sol[i] - hx*bcval[0];
				}
				res = alpha*sol[i] + 1.0/(hx*hx) * (2.0*sol[i] - solext - sol[i+1]);
			} else if (i == nx-1) {
				double solext = 0;
				if (bctype[1] == BCType_DIR) {
					solext = 2.0*bcval[1] - sol[i];
				} else if (bctype[1] == BCType_NEU) {
					solext = sol[i] + hx*bcval[1];
				}
				res = alpha*sol[i] + 1.0/(hx*hx) * (2.0*sol[i] - solext - sol[i-1]);
			} else {
				res = alpha*sol[i] + 1.0/(hx*hx) * (2.0*sol[i] - sol[i-1] - sol[i+1]);
			}
			res -= fs[i];


			ofs << xs[i] << "," 
				<< func_phi(xs[i]) << ","
				<< sol[i] << "," 
				<< err << ","
				<< res
				<< std::endl;
		}
	}


	fftw_free(xs);
	fftw_free(fs);
	fftw_free(rhs);
	fftw_free(fhat);
	fftw_free(sol);

	return 0;
}




int main2d(int argc, char *argv[]) {

	if (fftw_init_threads() == 0) {
		std::cerr << "FFTW failed to initialize thread model" << std::endl;
		exit(1);
	}

	int nthreads = omp_get_max_threads();
	//const int nthreads = 1;
	if (argc > 1) {
		nthreads = atoi(argv[1]);
	}
	std::cout << "Number of threads = " << nthreads << std::endl;

	const int ndim = 2;

	//
	//const double alpha = 1.0;
	const double alpha = 0.0;

	//const double xlo[ndim] = { 0.0, 0.0 };
	//const double xhi[ndim] = { 1.0, 1.0 };

	//const int ncell[ndim] = { 64, 64 };
	const int ncell[ndim] = { 512, 512 };
	//const int ncell[ndim] = { 1024, 1024 };
	const int nx = ncell[0];
	const int ny = ncell[1];

	const double bcpos[ndim][2] = {
		0.0, 1.0, // x low / high
		0.0, 1.0, // y low / high
	};

	const double cellsize[ndim] = {
		(bcpos[0][1] - bcpos[0][0]) / ncell[0],
		(bcpos[1][1] - bcpos[1][0]) / ncell[1],
	};
	const double hx = cellsize[0];
	const double hy = cellsize[1];

	const int bctype[ndim][2] = { 
		BCType_DIR, BCType_DIR, // x low / high
		BCType_DIR, BCType_DIR, // y low / high
	};

	double *cellpos[ndim];
	for (int dir=0; dir<ndim; dir++) {
		cellpos[dir] = (double*) fftw_malloc(sizeof(double) * ncell[dir]);
		for (int i=0; i<ncell[dir]; i++) {
			cellpos[dir][i] = bcpos[dir][0] + cellsize[dir]*(i+0.5);
		}
	}
	const double *xpos = cellpos[0];
	const double *ypos = cellpos[1];


	double* bcval[ndim][2];
	for (int dir=0; dir<ndim; dir++) {
		int nc = ncell[0]*ncell[1] / ncell[dir];
		for (int side=0; side<=1; side++) {
			bcval[dir][side] = (double*) fftw_malloc(sizeof(double) * nc);
		}
	}

	// set BC value
	for (int j=0; j<ny; j++) { // x low & high
		double y = ypos[j];
		bcval[0][0][j] = 5.0 * (y*y - y);
		bcval[0][1][j] = -y * pow(y-1,4);
	}
	for (int i=0; i<nx; i++) { // y low & high
		double x = xpos[i];
		bcval[1][0][i] = 0.5 * sin(6.0*M_PI*x);
		bcval[1][1][i] = sin(2.0*M_PI*x);
	}

	//
	double *fs = (double*) fftw_malloc(sizeof(double)*nx*ny);
	double *rhs = (double*) fftw_malloc(sizeof(double)*nx*ny);
	double *fhat = (double*) fftw_malloc(sizeof(double)*nx*ny);
	double *lambda = (double*) fftw_malloc(sizeof(double)*nx*ny);
	double *sol = (double*) fftw_malloc(sizeof(double)*nx*ny);

	{
		elaps_time = clock() - start_time;
		std::cout << "Setup time = " << elaps_time << std::endl;
		start_time = clock();
	}

	//
	fftw_plan plan_fwd = NULL;
	{
		fftw_r2r_kind xkind;
		double xnorm;
		selectTrans(true, nx, hx, bctype[0][0], bctype[0][1], xkind, xnorm);

		fftw_r2r_kind ykind;
		double ynorm;
		selectTrans(true, ny, hy, bctype[1][0], bctype[1][1], ykind, ynorm);

		fftw_plan_with_nthreads(nthreads);
		plan_fwd = fftw_plan_r2r_2d(nx, ny, rhs, fhat, xkind, ykind, FFTW_MEASURE);
	}

	//
	fftw_plan plan_bwd = NULL;
	{
		fftw_r2r_kind xkind;
		double xnorm;
		selectTrans(false, nx, hx, bctype[0][0], bctype[0][1], xkind, xnorm);

		fftw_r2r_kind ykind;
		double ynorm;
		selectTrans(false, ny, hy, bctype[1][0], bctype[1][1], ykind, ynorm);

		fftw_plan_with_nthreads(nthreads);
		plan_bwd = fftw_plan_r2r_2d(nx, ny, fhat, sol, xkind, ykind, FFTW_MEASURE);
	}

	{
		elaps_time = clock() - start_time;
		std::cout << "Plan time = " << elaps_time << std::endl;
		start_time = clock();
	}

	// set RHS, and take care of boundary values
	for (int i=0; i<nx; i++) {
		for (int j=0; j<ny; j++) {
			int idx = IDX2D(i,j);

			fs[idx] = -exp(xpos[i]*ypos[j]);
			rhs[idx] = fs[idx];

			if (i == 0) {
				fixRhs(nx, hx, rhs[idx], bctype[0][0], 0, -1.0, 0, bcval[0][0][j]);
			} else if (i == nx-1) {
				fixRhs(nx, hx, rhs[idx], bctype[0][1], 1, 1.0, 0, bcval[0][1][j]);
			}
			if (j == 0) {
				fixRhs(ny, hy, rhs[idx], bctype[1][0], 0, -1.0, 0, bcval[1][0][i]);
			} else if (j == ny-1) {
				fixRhs(ny, hy, rhs[idx], bctype[1][1], 1, 1.0, 0, bcval[1][1][i]);
			}
		}
	}


	// TODO compute perturbation
	double perturb = 0;
	if (alpha == 0) {
		//if (bctype[0]==BCType_NEU && bctype[1]==BCType_NEU) {
		//	perturb = 0;
		//	for (int i=0; i<nx; i++) {
		//		perturb += rhs[i];
		//	}
		//	perturb /= nx;
		//	for (int i=0; i<nx; i++) {
		//		rhs[i] -= perturb;
		//	}
		//}
	}
	std::cout << "PERTURB=" << perturb << std::endl;

	{
		elaps_time = clock() - start_time;
		std::cout << "BC time = " << elaps_time << std::endl;
		start_time = clock();
	}

	{
		fftw_r2r_kind xkind;
		double xnorm;
		selectTrans(true, nx, hx, bctype[0][0], bctype[0][1], xkind, xnorm);

		fftw_r2r_kind ykind;
		double ynorm;
		selectTrans(true, ny, hy, bctype[1][0], bctype[1][1], ykind, ynorm);

		if (0) {
			fftw_plan_with_nthreads(nthreads);
			fftw_plan plan = fftw_plan_r2r_2d(nx, ny, rhs, fhat, xkind, ykind, FFTW_ESTIMATE);
			fftw_execute(plan);
			fftw_destroy_plan(plan);
		} else {
			fftw_execute(plan_fwd);
		}

		// regularize
		for(int i=0; i<nx; i++) {
			for (int j=0; j<ny; j++) {
				int idx = IDX2D(i,j);
				fhat[idx] *= (xnorm * ynorm);
			}
		}
	}

	{
		elaps_time = clock() - start_time;
		std::cout << "Fwd-Tx time = " << elaps_time << std::endl;
		start_time = clock();
	}


	{
		//const double omega = M_PI / (2*nx);
		//for(int i=0; i<nx; i++) {
		//	//double lambda = 4.0 * pow(sin(omega*(i+1)), 2);
		//	double lambda = 2.0 - 2.0*cos(M_PI*(i+1)/nx);
		//	fhat[i] = fhat[i] / (lambda/hx2 + alpha);
		//}

		//
		computeLambda(lambda, nx, hx, bctype[0], ny, hy, bctype[1]);

		//
		for (int i=0; i<nx; i++) {
			for (int j=0; j<ny; j++) {
				int idx = IDX2D(i,j);

				if (lambda[idx]==0 && alpha==0) {
					fhat[idx] = 0;
				} else {
					fhat[idx] = fhat[idx] / (lambda[idx] + alpha);
				}
			}
		}

	}

	{
		elaps_time = clock() - start_time;
		std::cout << "Diag. time = " << elaps_time << std::endl;
		start_time = clock();
	}

	{
		// transform back to physical space

		fftw_r2r_kind xkind;
		double xnorm;
		selectTrans(false, nx, hx, bctype[0][0], bctype[0][1], xkind, xnorm);

		fftw_r2r_kind ykind;
		double ynorm;
		selectTrans(false, ny, hy, bctype[1][0], bctype[1][1], ykind, ynorm);

		if (0) {
			fftw_plan_with_nthreads(nthreads);
			fftw_plan plan = fftw_plan_r2r_2d(nx, ny, fhat, sol, xkind, ykind, FFTW_ESTIMATE);
			fftw_execute(plan);
			fftw_destroy_plan(plan);
		} else {
			fftw_execute(plan_bwd);
		}

		// regularize
		for(int i=0; i<nx; i++) {
			for (int j=0; j<ny; j++) {
				int idx = IDX2D(i,j);
				sol[idx] *= (xnorm * ynorm);
			}
		}
	}

	{
		elaps_time = clock() - start_time;
		std::cout << "Bwd-Tx time = " << elaps_time << std::endl;
		start_time = clock();
	}

	if (0) {
		std::ofstream ofs("../test/hoge.csv");
		ofs << "x,y,z,ana,fft,err,res" << std::endl;

		for (int j=0; j<ny; j++) {
			for (int i=0; i<nx; i++) {
				int idx = IDX2D(i,j);

				//double ana = func_phi(xs[i]);
				//double err = sol[i]/ana - 1;
				double ana = 0;
				double err = 0;

				// compute residual
				double res = 0;
				//if (i == 0) {
				//	double solext = 0;
				//	if (bctype[0] == BCType_DIR) {
				//		solext = 2.0*bcval[0] - sol[i];
				//	} else if (bctype[0] == BCType_NEU) {
				//		solext = sol[i] - hx*bcval[0];
				//	}
				//	res = alpha*sol[i] + 1.0/(hx*hx) * (2.0*sol[i] - solext - sol[i+1]);
				//} else if (i == nx-1) {
				//	double solext = 0;
				//	if (bctype[1] == BCType_DIR) {
				//		solext = 2.0*bcval[1] - sol[i];
				//	} else if (bctype[1] == BCType_NEU) {
				//		solext = sol[i] + hx*bcval[1];
				//	}
				//	res = alpha*sol[i] + 1.0/(hx*hx) * (2.0*sol[i] - solext - sol[i-1]);
				//} else {
				//	res = alpha*sol[i] + 1.0/(hx*hx) * (2.0*sol[i] - sol[i-1] - sol[i+1]);
				//}
				//res -= fs[i];


				ofs << xpos[i] << "," << ypos[j] << "," << 0.0 << ","
					<< ana << "," << sol[idx] << "," 
					<< err << "," << res << std::endl;
			}
		}
	}

	{
		elaps_time = clock() - start_time;
		std::cout << "IO time = " << elaps_time << std::endl;
		start_time = clock();
	}

	fftw_destroy_plan(plan_fwd);
	fftw_destroy_plan(plan_bwd);

	return 0;
}

int main2d_periodic(int argc, char *argv[]) {

	if (fftw_init_threads() == 0) {
		std::cerr << "FFTW failed to initialize thread model" << std::endl;
		exit(1);
	}

	int nthreads = omp_get_max_threads();
	if (argc > 1) {
		nthreads = atoi(argv[1]);
	}
	std::cout << "Number of threads = " << nthreads << std::endl;

	const int ndim = 2;

	//
	const double alpha = 0.0;

	const double bcpos[ndim][2] = {
		0.0, 1.0, // x low / high
		0.0, 1.0, // y low / high
	};

	//const int ncell[ndim] = { 64, 64 };
	const int ncell[ndim] = { 128, 128 };
	//const int ncell[ndim] = { 512, 512 };
	//const int ncell[ndim] = { 1024, 1024 };
	const int nx = ncell[0];
	const int ny = ncell[1];

	const double cellsize[ndim] = {
		(bcpos[0][1] - bcpos[0][0]) / ncell[0],
		(bcpos[1][1] - bcpos[1][0]) / ncell[1],
	};
	const double hx = cellsize[0];
	const double hy = cellsize[1];

	const int bctype[ndim][2] = { 
		// x low / high
		BCType_DIR, BCType_DIR, 
		//BCType_PER, BCType_PER, 
		// y low / high
		//BCType_DIR, BCType_DIR, 
		BCType_PER, BCType_PER, 
	};

	double *cellpos[ndim];
	for (int dir=0; dir<ndim; dir++) {
		cellpos[dir] = (double*) fftw_malloc(sizeof(double) * ncell[dir]);
		for (int i=0; i<ncell[dir]; i++) {
			cellpos[dir][i] = bcpos[dir][0] + cellsize[dir]*(i+0.5);
		}
	}
	const double *xpos = cellpos[0];
	const double *ypos = cellpos[1];

	double* bcval[ndim][2];
	for (int dir=0; dir<ndim; dir++) {
		int nc = ncell[0]*ncell[1] / ncell[dir];
		for (int side=0; side<=1; side++) {
			bcval[dir][side] = (double*) fftw_malloc(sizeof(double) * nc);
		}
	}

	// set BC value
	// u = sin(2*PI*x) * sin(4*PI*y)
	for (int j=0; j<ny; j++) { // x low & high
		double y = ypos[j];
		bcval[0][0][j] = 0;
		bcval[0][1][j] = 0;
	}
	for (int i=0; i<nx; i++) { // y low & high
		double x = xpos[i];
		bcval[1][0][i] = 0;
		bcval[1][1][i] = 0;
	}

	//
	double *fs = (double*) fftw_malloc(sizeof(double)*nx*ny);
	double *rhs = (double*) fftw_malloc(sizeof(double)*nx*ny);
	double *rhsx = (double*) fftw_malloc(sizeof(double)*nx*ny);
	double *rhsy = (double*) fftw_malloc(sizeof(double)*nx*ny);
	double *fhat = (double*) fftw_malloc(sizeof(double)*nx*ny);
	double *lambdax = (double*) fftw_malloc(sizeof(double)*nx);
	double *lambday = (double*) fftw_malloc(sizeof(double)*ny);
	double *sol = (double*) fftw_malloc(sizeof(double)*nx*ny);

	{
		elaps_time = clock() - start_time;
		std::cout << "Setup time = " << elaps_time << std::endl;
		start_time = clock();
	}

	////
	//fftw_plan plan_fwd = NULL;
	//{
	//	fftw_r2r_kind xkind;
	//	double xnorm;
	//	selectTrans(true, nx, hx, bctype[0][0], bctype[0][1], xkind, xnorm);

	//	fftw_r2r_kind ykind;
	//	double ynorm;
	//	selectTrans(true, ny, hy, bctype[1][0], bctype[1][1], ykind, ynorm);

	//	fftw_plan_with_nthreads(nthreads);
	//	plan_fwd = fftw_plan_r2r_2d(nx, ny, rhs, fhat, xkind, ykind, FFTW_MEASURE);
	//}

	////
	//fftw_plan plan_bwd = NULL;
	//{
	//	fftw_r2r_kind xkind;
	//	double xnorm;
	//	selectTrans(false, nx, hx, bctype[0][0], bctype[0][1], xkind, xnorm);

	//	fftw_r2r_kind ykind;
	//	double ynorm;
	//	selectTrans(false, ny, hy, bctype[1][0], bctype[1][1], ykind, ynorm);

	//	fftw_plan_with_nthreads(nthreads);
	//	plan_bwd = fftw_plan_r2r_2d(nx, ny, fhat, sol, xkind, ykind, FFTW_MEASURE);
	//}

	{
		elaps_time = clock() - start_time;
		std::cout << "Plan time = " << elaps_time << std::endl;
		start_time = clock();
	}

	// set RHS, and take care of boundary values
	for (int i=0; i<nx; i++) {
		for (int j=0; j<ny; j++) {
			int idx = IDX2D(i,j);

			fs[idx] = 20.0*M_PI*M_PI * sin(2.0*M_PI*xpos[i]) * sin(4.0*M_PI*ypos[j]);
			rhs[idx] = fs[idx];

			if (i == 0) {
				fixRhs(nx, hx, rhs[idx], bctype[0][0], 0, -1.0, 0, bcval[0][0][j]);
			} else if (i == nx-1) {
				fixRhs(nx, hx, rhs[idx], bctype[0][1], 1, 1.0, 0, bcval[0][1][j]);
			}
			if (j == 0) {
				fixRhs(ny, hy, rhs[idx], bctype[1][0], 0, -1.0, 0, bcval[1][0][i]);
			} else if (j == ny-1) {
				fixRhs(ny, hy, rhs[idx], bctype[1][1], 1, 1.0, 0, bcval[1][1][i]);
			}
		}
	}

	// TODO compute perturbation
	double perturb = 0;
	bool needPerturb = false;
	if (alpha == 0) {
		needPerturb = true;
		for (int dir=0; dir<ndim; dir++) {
			for (int side=0; side<=1; side++) {
				if (bctype[dir][side] == BCType_DIR) {
					// if having any DIR BC, no perturbation is needed
					needPerturb = false;
				}
			}
		}

		perturb = 0;
		if (needPerturb) {
			for (int i=0; i<nx; i++) {
				for (int j=0; j<ny; j++) {
					perturb += rhs[IDX2D(i,j)];
				}
			}
			perturb /= (nx*ny);
			for (int i=0; i<nx; i++) {
				for (int j=0; j<ny; j++) {
					rhs[IDX2D(i,j)] -= perturb;
				}
			}
		}
	}
	std::cout << "PERTURB=" << perturb << std::endl;

	{
		elaps_time = clock() - start_time;
		std::cout << "BC time = " << elaps_time << std::endl;
		start_time = clock();
	}

	{ // x forward, periodic
		fftw_r2r_kind xkind;
		double xnorm;
		selectTrans(true, nx, hx, bctype[0][0], bctype[0][1], xkind, xnorm);

		int rank = 1;
		int n[] = { nx };
		int howmany = ny;
		int idist = 1; 
		int odist = 1;
		int istride = ny;
		int ostride = ny;
		int *inembed = n;
		int *onembed = n;

		fftw_plan xplan = fftw_plan_many_r2r(rank, n, howmany, 
			rhs, inembed, istride, idist,
			rhsx, onembed, ostride, odist,
			&xkind, FFTW_ESTIMATE);
		fftw_execute(xplan);
		fftw_destroy_plan(xplan);

		// regularize
		for(int i=0; i<nx; i++) {
			for (int j=0; j<ny; j++) {
				int idx = IDX2D(i,j);
				rhsx[idx] *= (xnorm);
			}
		}

		// post-processing
		if (xkind == FFTW_R2HC) {
			int nxhalf = nx / 2;
			for (int j=0; j<ny; j++) {
				fhat[IDX2D(0,j)] = rhsx[IDX2D(0,j)];
				fhat[IDX2D(nx-1,j)] = rhsx[IDX2D(nxhalf,j)];
				for (int i=1; i<nxhalf; i++) {
					fhat[IDX2D(2*i-1,j)] = rhsx[IDX2D(i,j)];
					fhat[IDX2D(2*i,j)] = -rhsx[IDX2D(nx-i,j)];
				}
			}
		} else {
			for (int i=0; i<nx; i++) {
				for (int j=0; j<ny; j++) {
					int idx = IDX2D(i,j);
					fhat[idx] = rhsx[idx];
				}
			}
		}
	}

	{ // y forward
		fftw_r2r_kind ykind;
		double ynorm;
		selectTrans(true, ny, hy, bctype[1][0], bctype[1][1], ykind, ynorm);

		int rank = 1;
		int n[] = { ny };
		int howmany = nx;
		int idist = ny; 
		int odist = ny;
		int istride = 1;
		int ostride = 1;
		int *inembed = n;
		int *onembed = n;

		fftw_plan yplan = fftw_plan_many_r2r(rank, n, howmany,
			fhat, inembed, istride, idist,
			rhsy, onembed, ostride, odist,
			&ykind, FFTW_ESTIMATE);
		fftw_execute(yplan);
		fftw_destroy_plan(yplan);

		// regularize
		for(int i=0; i<nx; i++) {
			for (int j=0; j<ny; j++) {
				int idx = IDX2D(i,j);
				rhsy[idx] *= (ynorm);
			}
		}

		// post-processing
		if (ykind == FFTW_R2HC) {
			int nyhalf = ny / 2;
			for (int i=0; i<nx; i++) {
				fhat[IDX2D(i,0)] = rhsy[IDX2D(i,0)];
				fhat[IDX2D(i,ny-1)] = rhsy[IDX2D(i,nyhalf)];
				for (int j=1; j<nyhalf; j++) {
					fhat[IDX2D(i,2*j-1)] = rhsy[IDX2D(i,j)];
					fhat[IDX2D(i,2*j)] = -rhsy[IDX2D(i,ny-j)];
				}
			}
		} else { // copy to buffer
			for (int i=0; i<nx; i++) {
				for (int j=0; j<ny; j++) {
					int idx = IDX2D(i,j);
					fhat[idx] = rhsy[idx];
				}
			}
		}
	}

	{
		elaps_time = clock() - start_time;
		std::cout << "Fwd-Tx time = " << elaps_time << std::endl;
		start_time = clock();
	}


	{
		//
		computeLambda(nx, hx, bctype[0][0], bctype[0][1], lambdax);
		computeLambda(ny, hy, bctype[1][0], bctype[1][1], lambday);

		//
		for (int i=0; i<nx; i++) {
			for (int j=0; j<ny; j++) {
				int idx = IDX2D(i,j);

				const double eps = 1e-16;
				if (abs(lambdax[i]+lambday[j])<eps && alpha==0) {
					fhat[idx] = 0;
				} else {
					fhat[idx] = fhat[idx] / (lambdax[i] + lambday[j] + alpha);
				}
			}
		}

	}

	{
		elaps_time = clock() - start_time;
		std::cout << "Diag. time = " << elaps_time << std::endl;
		start_time = clock();
	}

	{ // y backward
		fftw_r2r_kind ykind;
		double ynorm;
		selectTrans(false, ny, hy, bctype[1][0], bctype[1][1], ykind, ynorm);

		// pre-processing
		if (ykind == FFTW_HC2R) {
			int nyhalf = ny / 2;
			for (int i=0; i<nx; i++) {
				rhsy[IDX2D(i,0)] = fhat[IDX2D(i,0)];
				rhsy[IDX2D(i,nyhalf)] = fhat[IDX2D(i,ny-1)];
				for (int j=1; j<nyhalf; j++) {
					rhsy[IDX2D(i,j)] = fhat[IDX2D(i,2*j-1)];
					rhsy[IDX2D(i,ny-j)] = -fhat[IDX2D(i,2*j)];
				}
			}
			for (int i=0; i<nx; i++) {
				for (int j=0; j<ny; j++) {
					int idx = IDX2D(i,j);
					fhat[idx] = rhsy[idx];
				}
			}
		} 

		int rank = 1;
		int n[] = { ny };
		int howmany = nx;
		int idist = ny; 
		int odist = ny;
		int istride = 1;
		int ostride = 1;
		int *inembed = n;
		int *onembed = n;

		fftw_plan yplan = fftw_plan_many_r2r(rank, n, howmany,
			fhat, inembed, istride, idist,
			rhsy, onembed, ostride, odist,
			&ykind, FFTW_ESTIMATE);
		fftw_execute(yplan);
		fftw_destroy_plan(yplan);

		// regularize
		for(int i=0; i<nx; i++) {
			for (int j=0; j<ny; j++) {
				int idx = IDX2D(i,j);
				rhsy[idx] *= (ynorm);
			}
		}
	}

	{ // x backward, periodic
		fftw_r2r_kind xkind;
		double xnorm;
		selectTrans(false, nx, hx, bctype[0][0], bctype[0][1], xkind, xnorm);

		// pre-processing
		if (xkind == FFTW_HC2R) {
			int nxhalf = nx / 2;
			for (int j=0; j<ny; j++) {
				fhat[IDX2D(0,j)] = rhsy[IDX2D(0,j)];
				fhat[IDX2D(nxhalf,j)] = rhsy[IDX2D(nx-1,j)];
				for (int i=1; i<nxhalf; i++) {
					fhat[IDX2D(i,j)] = rhsy[IDX2D(2*i-1,j)];
					fhat[IDX2D(nx-i,j)] = -rhsy[IDX2D(2*i,j)];
				}
			}
		} else {
			for (int i=0; i<nx; i++) {
				for (int j=0; j<ny; j++) {
					int idx = IDX2D(i,j);
					fhat[idx] = rhsy[idx];
				}
			}
		}

		int rank = 1;
		int n[] = { nx };
		int howmany = ny;
		int idist = 1; 
		int odist = 1;
		int istride = ny;
		int ostride = ny;
		int *inembed = n;
		int *onembed = n;

		fftw_plan xplan = fftw_plan_many_r2r(rank, n, howmany, 
			fhat, inembed, istride, idist,
			sol, onembed, ostride, odist,
			&xkind, FFTW_ESTIMATE);
		fftw_execute(xplan);
		fftw_destroy_plan(xplan);

		// regularize
		for(int i=0; i<nx; i++) {
			for (int j=0; j<ny; j++) {
				int idx = IDX2D(i,j);
				sol[idx] *= (xnorm);
			}
		}
	}

	{
		elaps_time = clock() - start_time;
		std::cout << "Bwd-Tx time = " << elaps_time << std::endl;
		start_time = clock();
	}

	if (1) {
		std::ofstream ofs("../test/hoge00.csv");
		ofs << "x,y,z,ana,fft,err,res" << std::endl;

		for (int j=0; j<ny; j++) {
			for (int i=0; i<nx; i++) {
				int idx = IDX2D(i,j);

				//double ana = func_phi(xs[i]);
				//double err = sol[i]/ana - 1;
				double ana = 0;
				double err = 0;

				// compute residual
				double res = 0;
				//if (i == 0) {
				//	double solext = 0;
				//	if (bctype[0] == BCType_DIR) {
				//		solext = 2.0*bcval[0] - sol[i];
				//	} else if (bctype[0] == BCType_NEU) {
				//		solext = sol[i] - hx*bcval[0];
				//	}
				//	res = alpha*sol[i] + 1.0/(hx*hx) * (2.0*sol[i] - solext - sol[i+1]);
				//} else if (i == nx-1) {
				//	double solext = 0;
				//	if (bctype[1] == BCType_DIR) {
				//		solext = 2.0*bcval[1] - sol[i];
				//	} else if (bctype[1] == BCType_NEU) {
				//		solext = sol[i] + hx*bcval[1];
				//	}
				//	res = alpha*sol[i] + 1.0/(hx*hx) * (2.0*sol[i] - solext - sol[i-1]);
				//} else {
				//	res = alpha*sol[i] + 1.0/(hx*hx) * (2.0*sol[i] - sol[i-1] - sol[i+1]);
				//}
				//res -= fs[i];


				ofs << xpos[i] << "," << ypos[j] << "," << 0.0 << ","
					<< ana << "," << sol[idx] << "," 
					<< err << "," << res << std::endl;
			}
		}
	}

	{
		elaps_time = clock() - start_time;
		std::cout << "IO time = " << elaps_time << std::endl;
		start_time = clock();
	}

	//fftw_destroy_plan(plan_fwd);
	//fftw_destroy_plan(plan_bwd);

	return 0;
} // main2d_periodic


int main3d(int argc, char *argv[]) {

	if (fftw_init_threads() == 0) {
		std::cerr << "FFTW failed to initialize thread model" << std::endl;
		exit(1);
	}

	int nthreads = omp_get_max_threads();
	if (argc > 1) {
		nthreads = atoi(argv[1]);
	}
	std::cout << "Number of threads = " << nthreads << std::endl;

	const int ndim = 3;

	//
	const double alpha = 0.0;

	const double bcpos[ndim][2] = {
		0.0, 1.0, // x low / high
		0.0, 1.0, // y low / high
		0.0, 1.0, // z low / high
	};

	//const int ncell[ndim] = { 64, 64, 64 };
	const int ncell[ndim] = { 128, 128, 128 };
	//const int ncell[ndim] = { 256, 256, 256 };
	const int nx = ncell[0];
	const int ny = ncell[1];
	const int nz = ncell[2];
	const int ndof = nx*ny*nz;
	std::cout << "Number of cells = " << nx << " " << ny << " " << nz << std::endl;

	const double cellsize[ndim] = {
		(bcpos[0][1] - bcpos[0][0]) / ncell[0],
		(bcpos[1][1] - bcpos[1][0]) / ncell[1],
		(bcpos[2][1] - bcpos[2][0]) / ncell[2],
	};
	const double hx = cellsize[0];
	const double hy = cellsize[1];
	const double hz = cellsize[2];

	const int bctype[ndim][2] = { 
		// x low / high
		//BCType_DIR, BCType_DIR, 
		BCType_PER, BCType_PER, 
		// y low / high
		//BCType_DIR, BCType_DIR, 
		BCType_PER, BCType_PER, 
		// z low / high
		BCType_DIR, BCType_DIR, 
		//BCType_PER, BCType_PER, 
	};

	double *cellpos[ndim];
	for (int dir=0; dir<ndim; dir++) {
		cellpos[dir] = (double*) fftw_malloc(sizeof(double) * ncell[dir]);
		for (int i=0; i<ncell[dir]; i++) {
			cellpos[dir][i] = bcpos[dir][0] + cellsize[dir]*(i+0.5);
		}
	}
	const double *xpos = cellpos[0];
	const double *ypos = cellpos[1];
	const double *zpos = cellpos[2];

	double* bcval[ndim][2];
	for (int dir=0; dir<ndim; dir++) {
		int nc = ndof / ncell[dir];
		for (int side=0; side<=1; side++) {
			bcval[dir][side] = (double*) fftw_malloc(sizeof(double) * nc);
		}
	}

	// TODO set BC value
	// u = sin(2*PI*x) * sin(2*PI*y) * z*(1-z)
	for (int j=0; j<ny; j++) { // x low & high
		for (int k=0; k<nz; k++) {
			int idx = IDX2D(j,k);
			bcval[0][0][idx] = 0;
			bcval[0][1][idx] = 0;
		}
	}
	//for (int i=0; i<nx; i++) { // y low & high
	//	double x = xpos[i];
	//	bcval[1][0][i] = 0;
	//	bcval[1][1][i] = 0;
	//}

	//
	double *fs = (double*) fftw_malloc(sizeof(double)*ndof);
	double *fhat = (double*) fftw_malloc(sizeof(double)*ndof);

	double *rhs = (double*) fftw_malloc(sizeof(double)*ndof);
	double *rhsx = (double*) fftw_malloc(sizeof(double)*ndof);
	double *rhsy = (double*) fftw_malloc(sizeof(double)*ndof);
	double *rhsz = (double*) fftw_malloc(sizeof(double)*ndof);

	double *lambdax = (double*) fftw_malloc(sizeof(double)*nx);
	double *lambday = (double*) fftw_malloc(sizeof(double)*ny);
	double *lambdaz = (double*) fftw_malloc(sizeof(double)*nz);

	double *sol = (double*) fftw_malloc(sizeof(double)*ndof);

	{
		elaps_time = clock() - start_time;
		std::cout << "Setup time = " << elaps_time << std::endl;
		start_time = clock();
	}

	{
		elaps_time = clock() - start_time;
		std::cout << "Plan time = " << elaps_time << std::endl;
		start_time = clock();
	}

	// set RHS
	// TODO take care of boundary values
	for (int i=0; i<nx; i++) {
		for (int j=0; j<ny; j++) {
			for (int k=0; k<nz; k++) {
				int idx = IDX3D(i,j,k);

				double x = xpos[i];
				double y = ypos[j];
				double z = zpos[k];

				double sx = sin(2*M_PI*x);
				double sy = sin(2*M_PI*y);
				double cx = cos(2*M_PI*x);
				double cy = cos(2*M_PI*y);
				
				//fs[idx] = 8.0*M_PI*M_PI * sx*sy*z*(1-z) + 2.0*sx*sy;
				fs[idx] = 8.0*M_PI*M_PI * sx*cy*z*(1-z) + 2.0*sx*cy;
				rhs[idx] = fs[idx];

				//if (i == 0) {
				//	fixRhs(nx, hx, rhs[idx], bctype[0][0], 0, -1.0, 0, bcval[0][0][j]);
				//} else if (i == nx-1) {
				//	fixRhs(nx, hx, rhs[idx], bctype[0][1], 1, 1.0, 0, bcval[0][1][j]);
				//}
				//if (j == 0) {
				//	fixRhs(ny, hy, rhs[idx], bctype[1][0], 0, -1.0, 0, bcval[1][0][i]);
				//} else if (j == ny-1) {
				//	fixRhs(ny, hy, rhs[idx], bctype[1][1], 1, 1.0, 0, bcval[1][1][i]);
				//}
			}
		}
	}

	// compute perturbation
	double perturb = 0;
	bool needPerturb = false;
	if (alpha == 0) {
		needPerturb = true;
		for (int dir=0; dir<ndim; dir++) {
			for (int side=0; side<=1; side++) {
				if (bctype[dir][side] == BCType_DIR) {
					// if having any DIR BC, no perturbation is needed
					needPerturb = false;
				}
			}
		}

		perturb = 0;
		if (needPerturb) {
			for (int idx=0; idx<ndof; idx++) {
				perturb += rhs[idx];
			}
			perturb /= ndof;
			for (int idx=0; idx<ndof; idx++) {
				rhs[idx] -= perturb;
			}
		}
	}
	std::cout << "PERTURB=" << perturb << std::endl;

	{
		elaps_time = clock() - start_time;
		std::cout << "BC time = " << elaps_time << std::endl;
		start_time = clock();
	}

	{ // x forward: RHS->RHSX
		fftw_r2r_kind xkind;
		double xnorm;
		selectTrans(true, nx, hx, bctype[0][0], bctype[0][1], xkind, xnorm);

		int rank = 1;
		int n[] = { nx };
		int howmany = ny*nz;
		int idist = 1; 
		int odist = 1;
		int istride = ny*nz;
		int ostride = ny*nz;
		int *inembed = n;
		int *onembed = n;
		double *in = rhs;
		double *out = xkind==FFTW_R2HC ? rhsx : fhat;

		fftw_plan_with_nthreads(nthreads);
		fftw_plan xplan = fftw_plan_many_r2r(rank, n, howmany, 
			in, inembed, istride, idist,
			out, onembed, ostride, odist,
			&xkind, FFTW_ESTIMATE);
		fftw_execute(xplan);
		fftw_destroy_plan(xplan);

		{
			elaps_time = clock() - start_time;
			std::cout << "Fwd-FFT-X time = " << elaps_time << std::endl;
			start_time = clock();
		}

		// regularize
		for (int idx=0; idx<ndof; idx++) {
			out[idx] *= xnorm;
		}

		// post-processing
		if (xkind == FFTW_R2HC) {
			int nxhalf = nx / 2;
#pragma omp parallel for collapse(2)
			for (int j=0; j<ny; j++) {
				for (int k=0; k<nz; k++) {
					fhat[IDX3D(0,j,k)] = rhsx[IDX3D(0,j,k)];
					fhat[IDX3D(nx-1,j,k)] = rhsx[IDX3D(nxhalf,j,k)];
					for (int i=1; i<nxhalf; i++) {
						fhat[IDX3D(2*i-1,j,k)] = rhsx[IDX3D(i,j,k)];
						fhat[IDX3D(2*i,j,k)] = -rhsx[IDX3D(nx-i,j,k)];
					}
				}
			}
		}
		// transpose (i,j,k) -> (k,i,j)
		for (int i=0; i<nx; i++) {
			for (int j=0; j<ny; j++) {
				for (int k=0; k<nz; k++) {
					//int idx0 = k + j*nz + i*nz*ny;
					int idx = IDX3D(i,j,k);
					rhsx[idx] = fhat[idx];
				}
			}
		}
	}

	{
		elaps_time = clock() - start_time;
		std::cout << "Fwd-X time = " << elaps_time << std::endl;
		start_time = clock();
	}

	{ // y forward: RHSX->RHSY
		fftw_r2r_kind ykind;
		double ynorm;
		selectTrans(true, ny, hy, bctype[1][0], bctype[1][1], ykind, ynorm);

		double *in = rhsx;
		double *out = ykind==FFTW_R2HC ? fhat : rhsy;
		for (int i=0; i<nx; i++) {
			int rank = 1;
			int n[] = { ny };
			int howmany = nz;
			int idist = 1;
			int odist = 1;
			int istride = nz;
			int ostride = nz;
			int *inembed = n;
			int *onembed = n;

			fftw_plan_with_nthreads(nthreads);
			fftw_plan yplan = fftw_plan_many_r2r(rank, n, howmany,
				in+i*ny*nz, inembed, istride, idist,
				out+i*ny*nz, onembed, ostride, odist,
				&ykind, FFTW_ESTIMATE);
			fftw_execute(yplan);
			fftw_destroy_plan(yplan);
		}

		{
			elaps_time = clock() - start_time;
			std::cout << "Fwd-FFT-Y time = " << elaps_time << std::endl;
			start_time = clock();
		}

		// 
		for (int idx=0; idx<ndof; idx++) {
			out[idx] *= ynorm;
		}

		//
		if (ykind == FFTW_R2HC) {
			int nyhalf = ny / 2;
			for (int i=0; i<nx; i++) {
				for (int k=0; k<nz; k++) {
					rhsy[IDX3D(i,0,k)] = fhat[IDX3D(i,0,k)];
					rhsy[IDX3D(i,ny-1,k)] = fhat[IDX3D(i,nyhalf,k)];
					for (int j=1; j<nyhalf; j++) {
						rhsy[IDX3D(i,2*j-1,k)] = fhat[IDX3D(i,j,k)];
						rhsy[IDX3D(i,2*j,k)] = -fhat[IDX3D(i,ny-j,k)];
					}
				}
			}
		}
		
	}

	{
		elaps_time = clock() - start_time;
		std::cout << "Fwd-Y time = " << elaps_time << std::endl;
		start_time = clock();
	}

	{ // z forward: RHSY->RHSZ
		fftw_r2r_kind zkind;
		double znorm;
		selectTrans(true, nz, hz, bctype[2][0], bctype[2][1], zkind, znorm);

		int rank = 1;
		int n[] = { nz };
		int howmany = nx*ny;
		int idist = nz;
		int odist = nz;
		int istride = 1;
		int ostride = 1;
		int *inembed = n;
		int *onembed = n;
		double *in = rhsy;
		double *out = zkind==FFTW_R2HC ? fhat : rhsz;

		fftw_plan_with_nthreads(nthreads);
		fftw_plan zplan = fftw_plan_many_r2r(rank, n, howmany,
			in, inembed, istride, idist,
			out, onembed, ostride, odist,
			&zkind, FFTW_ESTIMATE);
		fftw_execute(zplan);
		fftw_destroy_plan(zplan);

		{
			elaps_time = clock() - start_time;
			std::cout << "Fwd-FFT-Z time = " << elaps_time << std::endl;
			start_time = clock();
		}

		// regularize
		for (int idx=0; idx<ndof; idx++) {
			out[idx] *= znorm;
		}

		// post-processing
		if (zkind == FFTW_R2HC) {
			int nzhalf = nz / 2;
			for (int i=0; i<nx; i++) {
				for (int j=0; j<ny; j++) {
					rhsz[IDX3D(i,j,0)] = fhat[IDX3D(i,j,0)];
					rhsz[IDX3D(i,j,nz-1)] = fhat[IDX3D(i,j,nzhalf)];
					for (int k=1; k<nzhalf; k++) {
						rhsz[IDX3D(i,j,2*k-1)] = fhat[IDX3D(i,j,k)];
						rhsz[IDX3D(i,j,2*k)] = -fhat[IDX3D(i,j,nz-k)];
					}
				}
			}
		}
	}

	{
		elaps_time = clock() - start_time;
		std::cout << "Fwd-Z time = " << elaps_time << std::endl;
		start_time = clock();
	}


	{
		// RHSZ->FHAT
		for (int idx=0; idx<ndof; idx++) {
			fhat[idx] = rhsz[idx];
		}

		//
		computeLambda(nx, hx, bctype[0][0], bctype[0][1], lambdax);
		computeLambda(ny, hy, bctype[1][0], bctype[1][1], lambday);
		computeLambda(nz, hz, bctype[2][0], bctype[2][1], lambdaz);

		//
		for (int i=0; i<nx; i++) {
			for (int j=0; j<ny; j++) {
				for (int k=0; k<nz; k++) {
					int idx = IDX3D(i,j,k);

					const double eps = 1e-16;
					double lambdas = lambdax[i] + lambday[j] + lambdaz[k];
					if (abs(lambdas)<eps && alpha==0) {
						fhat[idx] = 0;
					} else {
						fhat[idx] = fhat[idx] / (lambdas + alpha);
					}
				}
			}
		}
	}

	{
		elaps_time = clock() - start_time;
		std::cout << "Diag. time = " << elaps_time << std::endl;
		start_time = clock();
	}

	{ // z backward FHAT->RHSZ
		fftw_r2r_kind zkind;
		double znorm;
		selectTrans(false, nz, hz, bctype[2][0], bctype[2][1], zkind, znorm);

		// pre-processing
		if (zkind == FFTW_HC2R) {
			int nzhalf = nz / 2;
			for (int i=0; i<nx; i++) {
				for (int j=0; j<ny; j++) {
					rhsz[IDX3D(i,j,0)] = fhat[IDX3D(i,j,0)];
					rhsz[IDX3D(i,j,nzhalf)] = fhat[IDX3D(i,j,nz-1)];
					for (int k=1; k<nzhalf; k++) {
						rhsz[IDX3D(i,j,k)] = fhat[IDX3D(i,j,2*k-1)];
						rhsz[IDX3D(i,j,nz-k)] = -fhat[IDX3D(i,j,2*k)];
					}
				}
			}
			for (int i=0; i<nx; i++) {
				for (int j=0; j<ny; j++) {
					for (int k=0; k<nz; k++) {
						int idx = IDX3D(i,j,k);
						fhat[idx] = rhsz[idx];
					}
				}
			}
		} 

		{
			elaps_time = clock() - start_time;
			std::cout << "Bwd-Precond-Z time = " << elaps_time << std::endl;
			start_time = clock();
		}

		int rank = 1;
		int n[] = { nz };
		int howmany = nx*ny;
		int idist = nz;
		int odist = nz;
		int istride = 1;
		int ostride = 1;
		int *inembed = n;
		int *onembed = n;
		double *in = fhat;
		double *out = rhsz;

		fftw_plan_with_nthreads(nthreads);
		fftw_plan zplan = fftw_plan_many_r2r(rank, n, howmany,
			in, inembed, istride, idist,
			out, onembed, ostride, odist,
			&zkind, FFTW_ESTIMATE);
		fftw_execute(zplan);
		fftw_destroy_plan(zplan);

		// regularize
		for (int idx=0; idx<ndof; idx++) {
			rhsz[idx] *= znorm;
		}
	}
	{
		elaps_time = clock() - start_time;
		std::cout << "Bwd-Z time = " << elaps_time << std::endl;
		start_time = clock();
	}

	{ // y backward, Z
		fftw_r2r_kind ykind;
		double ynorm;
		selectTrans(false, ny, hy, bctype[1][0], bctype[1][1], ykind, ynorm);

		// pre-processing
		if (ykind == FFTW_HC2R) {
			int nyhalf = ny / 2;
			for (int i=0; i<nx; i++) {
				for (int k=0; k<nz; k++) {
					rhsy[IDX3D(i,0,k)] = rhsz[IDX3D(i,0,k)];
					rhsy[IDX3D(i,nyhalf,k)] = rhsz[IDX3D(i,ny-1,k)];
					for (int j=1; j<nyhalf; j++) {
						rhsy[IDX3D(i,j,k)] = rhsz[IDX3D(i,2*j-1,k)];
						rhsy[IDX3D(i,ny-j,k)] = -rhsz[IDX3D(i,2*j,k)];
					}
				}
			}
			for (int i=0; i<nx; i++) {
				for (int j=0; j<ny; j++) {
					for (int k=0; k<nz; k++) {
						int idx = IDX3D(i,j,k);
						rhsz[idx] = rhsy[idx];
					}
				}
			}
		}

		{
			elaps_time = clock() - start_time;
			std::cout << "Bwd-Precond-Y time = " << elaps_time << std::endl;
			start_time = clock();
		}

		double *in = rhsz;
		double *out = rhsy;
		for (int i=0; i<nx; i++) {
			int rank = 1;
			int n[] = { ny };
			int howmany = nz;
			int idist = 1;
			int odist = 1;
			int istride = nz;
			int ostride = nz;
			int *inembed = n;
			int *onembed = n;

			fftw_plan_with_nthreads(nthreads);
			fftw_plan yplan = fftw_plan_many_r2r(rank, n, howmany,
				in+i*ny*nz, inembed, istride, idist,
				out+i*ny*nz, onembed, ostride, odist,
				&ykind, FFTW_ESTIMATE);
			fftw_execute(yplan);
			fftw_destroy_plan(yplan);
		}

		// 
		for (int idx=0; idx<ndof; idx++) {
			rhsy[idx] *= ynorm;
		}
	}
	{
		elaps_time = clock() - start_time;
		std::cout << "Bwd-Y time = " << elaps_time << std::endl;
		start_time = clock();
	}

	{ // x backward 
		fftw_r2r_kind xkind;
		double xnorm;
		selectTrans(false, nx, hx, bctype[0][0], bctype[0][1], xkind, xnorm);

		// pre-processing
		if (xkind == FFTW_HC2R) {
			int nxhalf = nx / 2;
			for (int j=0; j<ny; j++) {
				for (int k=0; k<nz; k++) {
					rhsx[IDX3D(0,j,k)] = rhsy[IDX3D(0,j,k)];
					rhsx[IDX3D(nxhalf,j,k)] = rhsy[IDX3D(nx-1,j,k)];
					for (int i=1; i<nxhalf; i++) {
						rhsx[IDX3D(i,j,k)] = rhsy[IDX3D(2*i-1,j,k)];
						rhsx[IDX3D(nx-i,j,k)] = -rhsy[IDX3D(2*i,j,k)];
					}
				}
			}
			for (int i=0; i<nx; i++) {
				for (int j=0; j<ny; j++) {
					for (int k=0; k<nz; k++) {
						int idx = IDX3D(i,j,k);
						rhsy[idx] = rhsx[idx];
					}
				}
			}
		}

		{
			elaps_time = clock() - start_time;
			std::cout << "Bwd-Precond-X time = " << elaps_time << std::endl;
			start_time = clock();
		}

		int rank = 1;
		int n[] = { nx };
		int howmany = ny*nz;
		int idist = 1; 
		int odist = 1;
		int istride = ny*nz;
		int ostride = ny*nz;
		int *inembed = n;
		int *onembed = n;
		double *in = rhsy;
		double *out = rhsx;

		fftw_plan_with_nthreads(nthreads);
		fftw_plan xplan = fftw_plan_many_r2r(rank, n, howmany, 
			in, inembed, istride, idist,
			out, onembed, ostride, odist,
			&xkind, FFTW_ESTIMATE);
		fftw_execute(xplan);
		fftw_destroy_plan(xplan);

		// regularize
		for (int idx=0; idx<ndof; idx++) {
			rhsx[idx] *= xnorm;
		}
	}

	{
		elaps_time = clock() - start_time;
		std::cout << "Bwd-X time = " << elaps_time << std::endl;
		start_time = clock();
	}

	{ // solution
		for (int idx=0; idx<ndof; idx++) {
			sol[idx] = rhsx[idx];
		}
	}

	if (0) {
		std::ofstream ofs("../test/hoge00.csv");
		ofs << "x,y,z,fft" << std::endl;

		for (int k=0; k<nz; k++) {
			for (int j=0; j<ny; j++) {
				for (int i=0; i<nx; i++) {
					int idx = IDX3D(i,j,k);

					double ana = 0;
					double err = 0;
					double res = 0;

					ofs << xpos[i] << "," << ypos[j] << "," << zpos[k] << ","
						<< sol[idx] << std::endl;
				}
			}
		}
	}

	{
		elaps_time = clock() - start_time;
		std::cout << "IO time = " << elaps_time << std::endl;
		start_time = clock();
	}

	//fftw_destroy_plan(plan_fwd);
	//fftw_destroy_plan(plan_bwd);

	return 0;
} // main3d



/// (i1,i2,i3) -> (i2,i3,i1)
void rotateIndiceFwd(double *out, const double *in, const int n1, const int n2, const int n3) {
#	pragma omp parallel for collapse(2) 
	for (int i2=0; i2<n2; i2++) {
		for (int i3=0; i3<n3; i3++) {
			for (int i1=0; i1<n1; i1++) {
				out[i2*n3*n1+i3*n1+i1] = in[i1*n2*n3+i2*n3+i3];
			}
		}
	}
}
/// (i1,i2,i3) -> (i3,i1,i2)
void rotateIndiceBwd(double *out, const double *in, const int n1, const int n2, const int n3) {
#	pragma omp parallel for collapse(2)
	for (int i3=0; i3<n3; i3++) {
		for (int i1=0; i1<n1; i1++) {
			for (int i2=0; i2<n2; i2++) {
				out[i3*n1*n2+i1*n2+i2] = in[i1*n2*n3+i2*n3+i3];
			}
		}
	}
}

/// perform FFT on index N3
void rfft3dLastIndex(fftw_r2r_kind kind, double norm,
	double *in, double *out, int n1, int n2, int n3,
	int nthreads=1) 
{
	int rank = 1;
	int n[] = { n3 };
	int howmany = n1*n2;
	int idist = n3;
	int odist = n3;
	int istride = 1; // continuous in memory
	int ostride = 1;
	int *inembed = n;
	int *onembed = n;

	fftw_plan_with_nthreads(nthreads);
	fftw_plan plan3 = fftw_plan_many_r2r(
		rank, n, howmany,
		in, inembed, istride, idist,
		out, onembed, ostride, odist,
		&kind, FFTW_ESTIMATE);
	
	if (!plan3) {
		std::cerr << __FUNCTION__ << ": failed to create FFTW plan" << std::endl;
		exit(1);
	}

	fftw_execute(plan3);
	fftw_destroy_plan(plan3);

#pragma omp parallel for
	for (int idx=0; idx<n1*n2*n3; idx++) {
		out[idx] *= norm;
	}
}

/// after R2HC, rearrange half-complex to cosine/sine for the N3 index
void rfft3dPostR2HCLastIndex(const double *in, double *out, int n1, int n2, int n3) {
	int n3half = n3 / 2;

#	pragma omp parallel for collapse(2)
	for (int i=0; i<n1; i++) {
		for (int j=0; j<n2; j++) {
			int offset = i*n2*n3 + j*n3;
			//
			out[offset] = in[offset];
			out[offset + n3-1] = in[offset + n3half];
			//
			for (int k=1; k<n3half; k++) {
				out[offset + 2*k-1] = in[offset + k];
				out[offset + 2*k] = -in[offset + n3-k];
			}

			//rhsz[IDX3D(i,j,0)] = fhat[IDX3D(i,j,0)];
			//rhsz[IDX3D(i,j,nz-1)] = fhat[IDX3D(i,j,nzhalf)];
			//for (int k=1; k<nzhalf; k++) {
			//	rhsz[IDX3D(i,j,2*k-1)] = fhat[IDX3D(i,j,k)];
			//	rhsz[IDX3D(i,j,2*k)] = -fhat[IDX3D(i,j,nz-k)];
			//}
		}
	}
}

/// before HC2R, rearrange cosine/sine to half-complex for the N3 index
void rfft3dPreHC2RLastIndex(const double *in, double *out, int n1, int n2, int n3) {
	int n3half = n3 / 2;

#	pragma omp parallel for collapse(2)
	for (int i=0; i<n1; i++) {
		for (int j=0; j<n2; j++) {
			int offset = i*n2*n3 + j*n3;
			//
			out[offset] = in[offset];
			out[offset + n3half] = in[offset + n3-1];
			//
			for (int k=1; k<n3half; k++) {
				out[offset + k] = in[offset + 2*k-1];
				out[offset + n3-k] = -in[offset + 2*k];
			}

			//rhsz[IDX3D(i,j,0)] = fhat[IDX3D(i,j,0)];
			//rhsz[IDX3D(i,j,nzhalf)] = fhat[IDX3D(i,j,nz-1)];
			//for (int k=1; k<nzhalf; k++) {
			//	rhsz[IDX3D(i,j,k)] = fhat[IDX3D(i,j,2*k-1)];
			//	rhsz[IDX3D(i,j,nz-k)] = -fhat[IDX3D(i,j,2*k)];
			//}
		}
	}
}


int main3d1(int argc, char *argv[]) {

	if (fftw_init_threads() == 0) {
		std::cerr << "FFTW failed to initialize thread model" << std::endl;
		exit(1);
	}

	int nthreads = omp_get_max_threads();
	if (argc > 1) {
		nthreads = atoi(argv[1]);
	}
	std::cout << "Number of threads = " << nthreads << std::endl;

	const int ndim = 3;

	//
	const double alpha = 0.0;

	const double bcpos[ndim][2] = {
		0.0, 1.0, // x low / high
		0.0, 1.0, // y low / high
		0.0, 1.0, // z low / high
	};

	//const int ncell[ndim] = { 64, 64, 64 };
	const int ncell[ndim] = { 128, 128, 128 };
	//const int ncell[ndim] = { 256, 256, 256 };
	const int nx = ncell[0];
	const int ny = ncell[1];
	const int nz = ncell[2];
	const int ndof = nx*ny*nz;
	std::cout << "Number of cells = " << nx << " " << ny << " " << nz << std::endl;

	const double cellsize[ndim] = {
		(bcpos[0][1] - bcpos[0][0]) / ncell[0],
		(bcpos[1][1] - bcpos[1][0]) / ncell[1],
		(bcpos[2][1] - bcpos[2][0]) / ncell[2],
	};
	const double hx = cellsize[0];
	const double hy = cellsize[1];
	const double hz = cellsize[2];

	const int bctype[ndim][2] = { 
		// x low / high
		//BCType_DIR, BCType_DIR, 
		BCType_PER, BCType_PER, 
		// y low / high
		//BCType_DIR, BCType_DIR, 
		BCType_PER, BCType_PER, 
		// z low / high
		BCType_DIR, BCType_DIR, 
		//BCType_PER, BCType_PER, 
	};

	double *cellpos[ndim];
	for (int dir=0; dir<ndim; dir++) {
		cellpos[dir] = (double*) fftw_malloc(sizeof(double) * ncell[dir]);
		for (int i=0; i<ncell[dir]; i++) {
			cellpos[dir][i] = bcpos[dir][0] + cellsize[dir]*(i+0.5);
		}
	}
	const double *xpos = cellpos[0];
	const double *ypos = cellpos[1];
	const double *zpos = cellpos[2];

	double* bcval[ndim][2];
	for (int dir=0; dir<ndim; dir++) {
		int nc = ndof / ncell[dir];
		for (int side=0; side<=1; side++) {
			bcval[dir][side] = (double*) fftw_malloc(sizeof(double) * nc);
		}
	}

	// TODO set BC value
	// u = sin(2*PI*x) * sin(2*PI*y) * z*(1-z)
	for (int j=0; j<ny; j++) { // x low & high
		for (int k=0; k<nz; k++) {
			int idx = IDX2D(j,k);
			bcval[0][0][idx] = 0;
			bcval[0][1][idx] = 0;
		}
	}


	//
	double *fs = (double*) fftw_malloc(sizeof(double)*ndof);
	double *fhat = (double*) fftw_malloc(sizeof(double)*ndof);
	double *ftmp = (double*) fftw_malloc(sizeof(double)*ndof);

	double *rhs = (double*) fftw_malloc(sizeof(double)*ndof);
	double *rhsx = (double*) fftw_malloc(sizeof(double)*ndof);
	double *rhsy = (double*) fftw_malloc(sizeof(double)*ndof);
	double *rhsz = (double*) fftw_malloc(sizeof(double)*ndof);

	double *lambdax = (double*) fftw_malloc(sizeof(double)*nx);
	double *lambday = (double*) fftw_malloc(sizeof(double)*ny);
	double *lambdaz = (double*) fftw_malloc(sizeof(double)*nz);

	double *sol = (double*) fftw_malloc(sizeof(double)*ndof);

	{
		elaps_time = clock() - start_time;
		std::cout << "Setup time = " << elaps_time << std::endl;
		start_time = clock();
	}

	{
		elaps_time = clock() - start_time;
		std::cout << "Plan time = " << elaps_time << std::endl;
		start_time = clock();
	}

	// set RHS
	// TODO take care of boundary values
	for (int i=0; i<nx; i++) {
		for (int j=0; j<ny; j++) {
			for (int k=0; k<nz; k++) {
				int idx = IDX3D(i,j,k);

				double x = xpos[i];
				double y = ypos[j];
				double z = zpos[k];

				double sx = sin(2*M_PI*x);
				double sy = sin(2*M_PI*y);
				double cx = cos(2*M_PI*x);
				double cy = cos(2*M_PI*y);
				
				fs[idx] = 8.0*M_PI*M_PI * sx*sy*z*(1-z) + 2.0*sx*sy;
				//fs[idx] = 8.0*M_PI*M_PI * sx*cy*z*(1-z) + 2.0*sx*cy;
				rhs[idx] = fs[idx];

				//if (i == 0) {
				//	fixRhs(nx, hx, rhs[idx], bctype[0][0], 0, -1.0, 0, bcval[0][0][j]);
				//} else if (i == nx-1) {
				//	fixRhs(nx, hx, rhs[idx], bctype[0][1], 1, 1.0, 0, bcval[0][1][j]);
				//}
				//if (j == 0) {
				//	fixRhs(ny, hy, rhs[idx], bctype[1][0], 0, -1.0, 0, bcval[1][0][i]);
				//} else if (j == ny-1) {
				//	fixRhs(ny, hy, rhs[idx], bctype[1][1], 1, 1.0, 0, bcval[1][1][i]);
				//}
			}
		}
	}

	// compute perturbation
	double perturb = 0;
	bool needPerturb = false;
	if (alpha == 0) {
		needPerturb = true;
		for (int dir=0; dir<ndim; dir++) {
			for (int side=0; side<=1; side++) {
				if (bctype[dir][side] == BCType_DIR) {
					// if having any DIR BC, no perturbation is needed
					needPerturb = false;
				}
			}
		}

		perturb = 0;
		if (needPerturb) {
			for (int idx=0; idx<ndof; idx++) {
				perturb += rhs[idx];
			}
			perturb /= ndof;
			for (int idx=0; idx<ndof; idx++) {
				rhs[idx] -= perturb;
			}
		}
	}
	std::cout << "PERTURB=" << perturb << std::endl;

	{
		elaps_time = clock() - start_time;
		std::cout << "BC time = " << elaps_time << std::endl;
		start_time = clock();
	}

	{ // x forward: RHS->RHSX
		// FWD rotate RHS(i,j,k) -> FHAT(j,k,i)
		rotateIndiceFwd(fhat, rhs, nx, ny, nz);
		{
			elaps_time = clock() - start_time;
			std::cout << "Fwd-Pre-X time = " << elaps_time << std::endl;
			start_time = clock();
		}

		fftw_r2r_kind xkind;
		double xnorm;
		selectTrans(true, nx, hx, bctype[0][0], bctype[0][1], xkind, xnorm);

		// FFT FHAT(j,k,i) -> RHSX(j,k,i) 
		if (xkind == FFTW_R2HC) {
			rfft3dLastIndex(xkind, xnorm, fhat, ftmp, ny, nz, nx, nthreads);
		} else {
			rfft3dLastIndex(xkind, xnorm, fhat, rhsx, ny, nz, nx, nthreads);
		}
		{
			elaps_time = clock() - start_time;
			std::cout << "Fwd-FFT-X time = " << elaps_time << std::endl;
			start_time = clock();
		}

		if (xkind == FFTW_R2HC) {
			rfft3dPostR2HCLastIndex(ftmp, rhsx, ny, nz, nx);
		}
		{
			elaps_time = clock() - start_time;
			std::cout << "Fwd-Post-X time = " << elaps_time << std::endl;
			start_time = clock();
		}
	}

	{ // y forward: RHSX->RHSY
		// FWD rotate RHSX(j,k,i) -> FHAT(k,i,j)
		rotateIndiceFwd(fhat, rhsx, ny, nz, nx);
		{
			elaps_time = clock() - start_time;
			std::cout << "Fwd-Pre-Y time = " << elaps_time << std::endl;
			start_time = clock();
		}

		fftw_r2r_kind ykind;
		double ynorm;
		selectTrans(true, ny, hy, bctype[1][0], bctype[1][1], ykind, ynorm);

		// FFT FHAT(k,i,j) -> RHSY(k,i,j)
		if (ykind == FFTW_R2HC) {
			rfft3dLastIndex(ykind, ynorm, fhat, ftmp, nz, nx, ny, nthreads);
		} else {
			rfft3dLastIndex(ykind, ynorm, fhat, rhsy, nz, nx, ny, nthreads);
		}
		{
			elaps_time = clock() - start_time;
			std::cout << "Fwd-FFT-Y time = " << elaps_time << std::endl;
			start_time = clock();
		}

		if (ykind == FFTW_R2HC) {
			rfft3dPostR2HCLastIndex(ftmp, rhsy, nz, nx, ny);
		}
		{
			elaps_time = clock() - start_time;
			std::cout << "Fwd-Post-Y time = " << elaps_time << std::endl;
			start_time = clock();
		}
	}

	{ // z forward: RHSY->RHSZ
		// FWD rotate RHSY(k,i,j) -> FHAT(i,j,k)
		rotateIndiceFwd(fhat, rhsy, nz, nx, ny);
		{
			elaps_time = clock() - start_time;
			std::cout << "Fwd-Pre-Z time = " << elaps_time << std::endl;
			start_time = clock();
		}

		fftw_r2r_kind zkind;
		double znorm;
		selectTrans(true, nz, hz, bctype[2][0], bctype[2][1], zkind, znorm);

		// FFT FHAT(i,j,k) -> RHSZ(i,j,k)
		if (zkind == FFTW_R2HC) {
			rfft3dLastIndex(zkind, znorm, fhat, ftmp, nx, ny, nz, nthreads);
		} else {
			rfft3dLastIndex(zkind, znorm, fhat, rhsz, nx, ny, nz, nthreads);
		}
		{
			elaps_time = clock() - start_time;
			std::cout << "Fwd-FFT-Z time = " << elaps_time << std::endl;
			start_time = clock();
		}

		if (zkind == FFTW_R2HC) {
			rfft3dPostR2HCLastIndex(ftmp, rhsz, nx, ny, nz);
		}
		{
			elaps_time = clock() - start_time;
			std::cout << "Fwd-Post-Z time = " << elaps_time << std::endl;
			start_time = clock();
		}
	}

	{ // solve in eigenvalue space
		// eigenvalues
		computeLambda(nx, hx, bctype[0][0], bctype[0][1], lambdax);
		computeLambda(ny, hy, bctype[1][0], bctype[1][1], lambday);
		computeLambda(nz, hz, bctype[2][0], bctype[2][1], lambdaz);
		{
			elaps_time = clock() - start_time;
			std::cout << "Eigenvalue time = " << elaps_time << std::endl;
			start_time = clock();
		}

		// solve RHSZ(i,j,k) -> RHSZ(i,j,k)
#		pragma omp parallel for collapse(2)
		for (int i=0; i<nx; i++) {
			for (int j=0; j<ny; j++) {
				for (int k=0; k<nz; k++) {
					int idx = IDX3D(i,j,k);

					const double eps = 1e-16;
					double lambdas = lambdax[i] + lambday[j] + lambdaz[k];
					if (abs(lambdas)<eps && alpha==0) {
						rhsz[idx] = 0;
					} else {
						rhsz[idx] /= (lambdas + alpha);
					}
				}
			}
		}
		{
			elaps_time = clock() - start_time;
			std::cout << "Diag. solve time = " << elaps_time << std::endl;
			start_time = clock();
		}
	}
	

	{ // z backward, RHSZ->RHSY
		fftw_r2r_kind zkind;
		double znorm;
		selectTrans(false, nz, hz, bctype[2][0], bctype[2][1], zkind, znorm);

		if (zkind == FFTW_HC2R) {
			rfft3dPreHC2RLastIndex(rhsz, ftmp, nx, ny, nz);
		}
		{
			elaps_time = clock() - start_time;
			std::cout << "Bwd-Pre-Z time = " << elaps_time << std::endl;
			start_time = clock();
		}

		// FFT RHSZ(i,j,k) -> FHAT(i,j,k)
		if (zkind == FFTW_HC2R) {
			rfft3dLastIndex(zkind, znorm, ftmp, fhat, nx, ny, nz, nthreads);
		} else {
			rfft3dLastIndex(zkind, znorm, rhsz, fhat, nx, ny, nz, nthreads);
		}
		{
			elaps_time = clock() - start_time;
			std::cout << "Bwd-FFT-Z time = " << elaps_time << std::endl;
			start_time = clock();
		}

		// BWD rotate FHAT(i,j,k) -> RHSY(k,i,j)
		rotateIndiceBwd(rhsy, fhat, nx, ny, nz);
		{
			elaps_time = clock() - start_time;
			std::cout << "Bwd-Post-Z time = " << elaps_time << std::endl;
			start_time = clock();
		}
	}

	{ // y backward, RHSY->RHSX
		fftw_r2r_kind ykind;
		double ynorm;
		selectTrans(false, ny, hy, bctype[1][0], bctype[1][1], ykind, ynorm);

		if (ykind == FFTW_HC2R) {
			rfft3dPreHC2RLastIndex(rhsy, ftmp, nz, nx, ny);
		}
		{
			elaps_time = clock() - start_time;
			std::cout << "Bwd-Pre-Y time = " << elaps_time << std::endl;
			start_time = clock();
		}

		// FFT RHSY(k,i,j) -> FHAT(k,i,j)
		if (ykind == FFTW_HC2R) {
			rfft3dLastIndex(ykind, ynorm, ftmp, fhat, nz, nx, ny, nthreads);
		} else {
			rfft3dLastIndex(ykind, ynorm, rhsy, fhat, nz, nx, ny, nthreads);
		}
		{
			elaps_time = clock() - start_time;
			std::cout << "Bwd-FFT-Y time = " << elaps_time << std::endl;
			start_time = clock();
		}

		// BWD rotate FHAT(k,i,j) -> RHSX(j,k,i)
		rotateIndiceBwd(rhsx, fhat, nz, nx, ny);
		{
			elaps_time = clock() - start_time;
			std::cout << "Bwd-Post-Y time = " << elaps_time << std::endl;
			start_time = clock();
		}
	}

	{ // x backward, RHSX->SOL
		fftw_r2r_kind xkind;
		double xnorm;
		selectTrans(false, nx, hx, bctype[0][0], bctype[0][1], xkind, xnorm);

		if (xkind == FFTW_HC2R) {
			rfft3dPreHC2RLastIndex(rhsx, ftmp, ny, nz, nx);
		}
		{
			elaps_time = clock() - start_time;
			std::cout << "Bwd-Pre-X time = " << elaps_time << std::endl;
			start_time = clock();
		}

		// FFT RHSX(j,k,i) -> FHAT(j,k,i)
		if (xkind == FFTW_HC2R) {
			rfft3dLastIndex(xkind, xnorm, ftmp, fhat, ny, nz, nx, nthreads);
		} else {
			rfft3dLastIndex(xkind, xnorm, rhsx, fhat, ny, nz, nx, nthreads);
		}
		{
			elaps_time = clock() - start_time;
			std::cout << "Bwd-FFT-X time = " << elaps_time << std::endl;
			start_time = clock();
		}

		// BWD rotate FHAT(j,k,i) -> SOL(i,j,k)
		rotateIndiceBwd(sol, fhat, ny, nz, nx);
		{
			elaps_time = clock() - start_time;
			std::cout << "Bwd-Post-X time = " << elaps_time << std::endl;
			start_time = clock();
		}
	}

	if (0) {
		std::ofstream ofs("../test/hoge00.csv");
		ofs << "x,y,z,fft" << std::endl;

		for (int k=0; k<nz; k++) {
			for (int j=0; j<ny; j++) {
				for (int i=0; i<nx; i++) {
					int idx = IDX3D(i,j,k);

					double ana = 0;
					double err = 0;
					double res = 0;

					ofs << xpos[i] << "," << ypos[j] << "," << zpos[k] << ","
						<< sol[idx] << std::endl;
				}
			}
		}
	}

	{
		elaps_time = clock() - start_time;
		std::cout << "IO time = " << elaps_time << std::endl;
		start_time = clock();
	}
	return 0;
} // main3d1

#define VTKHELPER_LINKER_PRAGMA
#include <vtkHelper.hpp>
#include <vtkRectilinearGrid.h>
#include <vtkXMLRectilinearGridWriter.h>
#include <vtkCellData.h>

int main3d_solver(int argc, char *argv[]) {
	const int ndim = 3;

	//
	const double alpha = 0.0;

	const double bcpos[ndim][2] = {
		0.0, 1.0, // x low / high
		0.0, 1.0, // y low / high
		0.0, 1.0, // z low / high
	};

	//const int ncell[ndim] = { 16, 16, 16 };
	//const int ncell[ndim] = { 32, 32, 32 };
	const int ncell[ndim] = { 64, 64, 64 };
	//const int ncell[ndim] = { 128, 128, 128 };
	//const int ncell[ndim] = { 256, 256, 256 };

	//const int ncell[ndim] = { 50, 50, 50 };

	const int nx = ncell[0];
	const int ny = ncell[1];
	const int nz = ncell[2];
	const int ndof = nx*ny*nz;
	std::cout << "Number of cells = " << nx << " " << ny << " " << nz << std::endl;

	const double cellsize[ndim] = {
		(bcpos[0][1] - bcpos[0][0]) / ncell[0],
		(bcpos[1][1] - bcpos[1][0]) / ncell[1],
		(bcpos[2][1] - bcpos[2][0]) / ncell[2],
	};
	const double hx = cellsize[0];
	const double hy = cellsize[1];
	const double hz = cellsize[2];

	const int bctype[ndim][2] = { 
		// x low / high
		BCType_DIR, BCType_DIR, 
		//BCType_NEU, BCType_NEU,
		//BCType_PER, BCType_PER, 
		// y low / high
		BCType_DIR, BCType_DIR, 
		//BCType_NEU, BCType_NEU,
		//BCType_PER, BCType_PER, 
		// z low / high
		BCType_DIR, BCType_DIR,
		//BCType_NEU, BCType_NEU,
		//BCType_PER, BCType_PER, 
	};

	double *cellpos[ndim];
	for (int dir=0; dir<ndim; dir++) {
		cellpos[dir] = (double*) fftw_malloc(sizeof(double) * ncell[dir]);
		for (int i=0; i<ncell[dir]; i++) {
			cellpos[dir][i] = bcpos[dir][0] + cellsize[dir]*(i+0.5);
		}
	}
	const double *xpos = cellpos[0];
	const double *ypos = cellpos[1];
	const double *zpos = cellpos[2];

	double* bcval[ndim][2];
	for (int dir=0; dir<ndim; dir++) {
		int nc = ndof / ncell[dir];
		for (int side=0; side<=1; side++) {
			bcval[dir][side] = (double*) fftw_malloc(sizeof(double) * nc);
		}
	}

	// TODO set BC value
	// u = sin(2*PI*x) * sin(2*PI*y) * z*(1-z)
	//for (int j=0; j<ny; j++) { // x low & high
	//	for (int k=0; k<nz; k++) {
	//		int idx = IDX2D(j,k);
	//		bcval[0][0][idx] = 0;
	//		bcval[0][1][idx] = 0;
	//	}
	//}

	//
	double *fs = (double*) fftw_malloc(sizeof(double)*ndof);
	double *rhs = (double*) fftw_malloc(sizeof(double)*ndof);
	double *sol = (double*) fftw_malloc(sizeof(double)*ndof);
	double *ana = (double*) fftw_malloc(sizeof(double)*ndof);
	double *res = (double*) fftw_malloc(sizeof(double)*ndof);

	// set RHS
	// TODO take care of boundary values
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
				ana[idx] = sx * sy * z*(1-z);
				rhs[idx] = 8.0*M_PI*M_PI * sx*sy*z*(1-z) + 2.0*sx*sy;

				// cos(2*pi*x) * cos(2*pi*y) * cos(2*pi*z)
				//fs[idx] = 12.0*M_PI*M_PI * cx*cy*cz;
				//ana[idx] = cx * cy * cz;
				//

				//fs[idx] = 4.0*M_PI*M_PI * (y*y*z*z + z*z*x*x + x*x*y*y) * sin(2.0*M_PI * x*y*z);
				//ana[idx] = sin(2.0*M_PI * x*y*z);

				//rhs[idx] = fs[idx];
				//if (i == 0) {
				//	double bcval = sin(2.0*M_PI * bcpos[0][0]*y*z);
				//	fixRhs(nx, hx, rhs[idx], bctype[0][0], 0, -1, 0, bcval);
				//} else if (i == nx-1) {
				//	double bcval = sin(2.0*M_PI * bcpos[0][1]*y*z);
				//	fixRhs(nx, hx, rhs[idx], bctype[0][1], 1, 1, 0, bcval);
				//}
				//if (j == 0) {
				//	double bcval = sin(2.0*M_PI * x*bcpos[1][0]*z);
				//	fixRhs(ny, hy, rhs[idx], bctype[1][0], 0, -1, 0, bcval);
				//} else if (j == ny-1) {
				//	double bcval = sin(2.0*M_PI * x*bcpos[1][1]*z);
				//	fixRhs(ny, hy, rhs[idx], bctype[1][1], 1, 1, 0, bcval);
				//}
				//if (k == 0) {
				//	double bcval = sin(2.0*M_PI * x*y*bcpos[2][0]);
				//	fixRhs(ny, hz, rhs[idx], bctype[2][0], 0, -1, 0, bcval);
				//} else if (k == nz-1) {
				//	double bcval = sin(2.0*M_PI * x*y*bcpos[2][1]);
				//	fixRhs(ny, hz, rhs[idx], bctype[2][1], 1, 1, 0, bcval);
				//}
			}
		}
	}

	{
		elaps_time = clock() - start_time;
		std::cout << "Setup time = " << elaps_time << std::endl;
		start_time = clock();
		//CLOCKS_PER_SEC
	}

	// define a Poisson solver
	//FastPoissonSolver3D solver(ncell, cellsize);
	//// set BC
	//for (int dir=0; dir<ndim; dir++) {
	//	solver.setBCType(dir, bctype[dir][0], bctype[dir][1]);
	//}

	FastPoissonSolver3D::PoissonProb prob;
	for (int dir=0; dir<3; dir++) {
		prob.cellnum[dir] = ncell[dir];
		prob.cellsize[dir] = cellsize[dir];
		
		prob.bctype[dir][0] = bctype[dir][0];
		prob.bctype[dir][1] = bctype[dir][1];
	}
	FastPoissonSolver3D solver(prob);

	std::cout << "is degenerate = " << solver.isDegenerate() << std::endl;

	// call this before using it
	solver.initialize();

	{
		elaps_time = clock() - start_time;
		std::cout << "Plan time = " << elaps_time << std::endl;
		start_time = clock();
	}

	
	

	const int ncycle = 16;
	for (int cycle=0; cycle<ncycle; cycle++) {
		// set RHS
		cblas_dcopy(ndof, rhs, 1, solver.getRhsData(), 1);

		// solve
		int ret = solver.solve();
		if (ret != 0) {
			std::cerr << "Poisson solver failed" << std::endl;
			exit(1);
		}

		// retrieve solution
		cblas_dcopy(ndof, solver.getSolData(), 1, sol, 1);
	}

	// residual
	for (int i=0; i<nx; i++) {
		for (int j=0; j<ny; j++) {
			for (int k=0; k<nz; k++) {
				int idx = IDX3D(i,j,k);

				res[idx] = 0;

				double lap = 0;
				double ul, ur;

				if (i == 0) {
					if (bctype[0][0] == BCType_PER) ul = sol[IDX3D(nx-1,j,k)];
					else if (bctype[0][0] == BCType_DIR) ul = -sol[idx];
					else if (bctype[0][0] == BCType_NEU) ul = sol[idx];
				} else {
					ul = sol[IDX3D(i-1,j,k)];
				}
				if (i == nx-1) {
					if (bctype[0][1] == BCType_PER) ur = sol[IDX3D(0,j,k)];
					else if (bctype[0][1] == BCType_DIR) ur = -sol[idx];
					else if (bctype[0][1] == BCType_NEU) ur = sol[idx];
				} else {
					ur = sol[IDX3D(i+1,j,k)];
				}
				lap += 1.0/(hx*hx) * (2.0*sol[idx] - ul - ur);

				if (j == 0) {
					if (bctype[1][0] == BCType_PER) ul = sol[IDX3D(i,ny-1,k)];
					else if (bctype[1][0] == BCType_DIR) ul = -sol[idx];
					else if (bctype[1][0] == BCType_NEU) ul = sol[idx];
				} else {
					ul = sol[IDX3D(i,j-1,k)];
				}
				if (j == ny-1) {
					if (bctype[1][1] == BCType_PER) ur = sol[IDX3D(i,0,k)];
					else if (bctype[1][1] == BCType_DIR) ur = -sol[idx];
					else if (bctype[1][1] == BCType_NEU) ur = sol[idx];
				} else {
					ur = sol[IDX3D(i,j+1,k)];
				}
				lap += 1.0/(hy*hy) * (2.0*sol[idx] - ul - ur);

				if (k == 0) {
					if (bctype[2][0] == BCType_PER) ul = sol[IDX3D(i,j,nz-1)];
					else if (bctype[2][0] == BCType_DIR) ul = -sol[idx];
					else if (bctype[2][0] == BCType_NEU) ul = sol[idx];
				} else {
					ul = sol[IDX3D(i,j,k-1)];
				}
				if (k == nz-1) {
					if (bctype[2][1] == BCType_PER) ur = sol[IDX3D(i,j,0)];
					else if (bctype[2][1] == BCType_DIR) ur = -sol[idx];
					else if (bctype[2][1] == BCType_NEU) ur = sol[idx];
				} else {
					ur = sol[IDX3D(i,j,k+1)];
				}
				lap += 1.0/(hz*hz) * (2.0*sol[idx] - ul - ur);

				res[idx] = alpha*sol[idx] + lap - rhs[idx];
			}
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
			vtkHelper_declareField<vtkDoubleArray>(grid, "fft", 1, ndof);
		vtkSmartPointer<vtkDoubleArray> anaArray = 
			vtkHelper_declareField<vtkDoubleArray>(grid, "ana", 1, ndof);
		vtkSmartPointer<vtkDoubleArray> resArray = 
			vtkHelper_declareField<vtkDoubleArray>(grid, "res", 1, ndof);

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
	return 0;
} // main3d_solver


/**
 * test node-based
 */
int main3d_solver2(int argc, char *argv[]) {
	const int ndim = 3;

	//
	const double alpha = 0.0;

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

	const int bctype[ndim][2] = { 
		// x low / high
		BCType_CDIR, BCType_CDIR,
		// y low / high
		BCType_CDIR, BCType_CDIR, 
		// z low / high
		BCType_CDIR, BCType_CDIR,
	};

	double *cellpos[ndim];
	for (int dir=0; dir<ndim; dir++) {
		cellpos[dir] = (double*) fftw_malloc(sizeof(double) * nnode[dir]);
		for (int i=0; i<nnode[dir]; i++) {
			cellpos[dir][i] = bcpos[dir][0] + cellsize[dir]*(i);
		}
	}
	const double *xpos = cellpos[0];
	const double *ypos = cellpos[1];
	const double *zpos = cellpos[2];

	//
	double *fs = (double*) fftw_malloc(sizeof(double)*nvar);
	double *rhs = (double*) fftw_malloc(sizeof(double)*nvar);
	double *sol = (double*) fftw_malloc(sizeof(double)*nvar);
	double *ana = (double*) fftw_malloc(sizeof(double)*nvar);
	double *res = (double*) fftw_malloc(sizeof(double)*nvar);

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

				// cos(2*pi*x) * cos(2*pi*y) * cos(2*pi*z)
				//fs[idx] = 12.0*M_PI*M_PI * cx*cy*cz;
				//ana[idx] = cx * cy * cz;
				//

				//fs[idx] = 4.0*M_PI*M_PI * (y*y*z*z + z*z*x*x + x*x*y*y) * sin(2.0*M_PI * x*y*z);
				//ana[idx] = sin(2.0*M_PI * x*y*z);

				rhs[idx] = fs[idx];
			}
		}
	}
	// TODO take care of boundary values
	if (1) {
	for (int i=0; i<nx; i++) {
		for (int j=0; j<ny; j++) {
			for (int k=0; k<nz; k++) {
				int idx = IDX3D(i,j,k);

				if (i == 1) {
					double bcval = ana[IDX3D(i-1,j,k)];
					fixRhs(nx, hx, rhs[idx], bctype[0][0], 0, -1, 0, bcval);
				} else if (i == nx-2) {
					double bcval = ana[IDX3D(i+1,j,k)];
					fixRhs(nx, hx, rhs[idx], bctype[0][1], 1, 1, 0, bcval);
				}
				if (j == 1) {
					double bcval = ana[IDX3D(i,j-1,k)];
					fixRhs(ny, hy, rhs[idx], bctype[1][0], 0, -1, 0, bcval);
				} else if (j == ny-2) {
					double bcval = ana[IDX3D(i,j+1,k)];
					fixRhs(ny, hy, rhs[idx], bctype[1][1], 1, 1, 0, bcval);
				}
				if (k == 1) {
					double bcval = ana[IDX3D(i,j,k-1)];
					fixRhs(nz, hz, rhs[idx], bctype[2][0], 0, -1, 0, bcval);
				} else if (k == nz-2) {
					double bcval = ana[IDX3D(i,j,k+1)];
					fixRhs(nz, hz, rhs[idx], bctype[2][1], 1, 1, 0, bcval);
				}
			}
		}
	}
	}
	{
		elaps_time = clock() - start_time;
		std::cout << "Setup time = " << elaps_time << std::endl;
		start_time = clock();
	}

	// define a Poisson solver
	FastPoissonSolver3D::PoissonProb prob;
	for (int dir=0; dir<3; dir++) {
		prob.cellnum[dir] = nnode[dir] - 2;
		prob.cellsize[dir] = cellsize[dir];
		
		prob.bctype[dir][0] = bctype[dir][0];
		prob.bctype[dir][1] = bctype[dir][1];
	}
	FastPoissonSolver3D solver(prob);

	std::cout << "is degenerate = " << solver.isDegenerate() << std::endl;

	// call this before using it
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

					res[idx] = alpha*sol[idx] + lap - rhs[idx];
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
			vtkHelper_declareField<vtkDoubleArray>(grid, "fft", 1, nvar);
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
	return 0;
} // main3d_solver2



int main(int argc, char *argv[]) {
	int ret = 0;

	start_time = clock();
	clock_t start_time0 = start_time;

	//ret = main2d(argc, argv);
	//ret = main2d_periodic(argc, argv);
	//ret = main3d(argc, argv);
	//ret = main3d1(argc, argv);
	ret = main3d_solver(argc, argv);
	//ret = main3d_solver2(argc, argv);

	elaps_time = clock() - start_time0;
	std::cout << "Elapsed = " << elaps_time << std::endl;

	return ret;
}



