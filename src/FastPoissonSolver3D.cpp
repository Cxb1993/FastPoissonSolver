

#include <iostream>
#include <fstream>

#include <cstdlib>
#include <ctime>

#define _USE_MATH_DEFINES
#include <math.h>

//#ifdef _OPENMP
#include <omp.h>
//#endif

#include <fftw3.h>
#pragma comment(lib, "libfftw3-3.lib")
//#pragma comment(lib, "libfftw-3.3.lib")

#include "FastPoisson.h"
#include "SolverUtil.h"


static void computeLambda(int n, double h, int bc_lo, int bc_hi, double *lambda) {
	const double h2inv = 1.0 / (h*h);

	if (bc_lo==BCType_PER && bc_hi==BCType_PER) {
		const double w = M_PI / n;
		lambda[0] = 0;
		// TODO check odd/even
		lambda[n-1] = 4.0 * h2inv;
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
	} else if (bc_lo==BCType_CDIR && bc_hi==BCType_CDIR) {
		//const double w = M_PI / (n+1);
		const double w = M_PI / (2*(n+1));
		for (int i=0; i<n; i++) {
			//lambda[i] = (2.0 - 2.0*cos(w*(i+1))) * h2inv;
			lambda[i] = 4.0*pow(sin(w*(i+1)), 2) * h2inv;
		}
	}
	else {
		std::cerr << __FUNCTION__ << ": Invalid BC pairing: "
			<< "LO=" << bc_lo << ", HI="<< bc_hi
			<< std::endl;
		exit(1);
	}
}

static void selectTrans(bool isfwd,
	int n, double h, int bc_lo, int bc_hi,
	fftw_r2r_kind &fftwkind, double &normalizer) 
{
	if (bc_lo==BCType_PER && bc_hi==BCType_PER) {
		if (isfwd) {
			fftwkind = FFTW_R2HC;
			normalizer = 1.0 / n;
		} else {
			fftwkind = FFTW_HC2R;
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
	} else if (bc_lo==BCType_CDIR && bc_hi==BCType_CDIR) {
		if (isfwd) {
			fftwkind = FFTW_RODFT00;
			normalizer = 1.0 / (n+1);
		} else {
			fftwkind = FFTW_RODFT00;
			normalizer = 0.5;
		}
	}
	else {
		std::cerr << __FUNCTION__ << ": Invalid BC pairing: "
			<< "LO=" << bc_lo << ", HI="<< bc_hi
			<< std::endl;
		exit(1);
	}
} // selectTrans


/**
 * Create FFT plan, 
 * will be performed on the N3 dimension for N1*N2 times
 */
static fftw_plan rfft3dPlanLastIndex(fftw_r2r_kind kind,
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
		&kind, FFTW_MEASURE);
	
	if (!plan3) {
		std::cerr << __FUNCTION__ 
			<< ": failed to create FFTW plan" 
			<< "; kind=" << (int) kind << std::endl;
		exit(1);
	}

	return plan3;
}

/// after R2HC, rearrange half-complex to cosine/sine for the N3 index
static void rfft3dPostR2HCLastIndex(const double *in, double *out, int n1, int n2, int n3) {
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
static void rfft3dPreHC2RLastIndex(const double *in, double *out, int n1, int n2, int n3) {
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


///// (i1,i2,i3) -> (i2,i3,i1)
//inline void rotateIndiceFwd(double *out, const double *in, const int n1, const int n2, const int n3) {
//#	pragma omp parallel for collapse(2) 
//	for (int i2=0; i2<n2; i2++) {
//		for (int i3=0; i3<n3; i3++) {
//			for (int i1=0; i1<n1; i1++) {
//				out[i2*n3*n1+i3*n1+i1] = in[i1*n2*n3+i2*n3+i3];
//			}
//		}
//	}
//}
///// (i1,i2,i3) -> (i3,i1,i2)
//inline void rotateIndiceBwd(double *out, const double *in, const int n1, const int n2, const int n3) {
//#	pragma omp parallel for collapse(2)
//	for (int i3=0; i3<n3; i3++) {
//		for (int i1=0; i1<n1; i1++) {
//			for (int i2=0; i2<n2; i2++) {
//				out[i3*n1*n2+i1*n2+i2] = in[i1*n2*n3+i2*n3+i3];
//			}
//		}
//	}
//}


FastPoissonSolver3D::FastPoissonSolver3D(const PoissonProb &prob)
	: m_alpha(0.0), m_beta(1.0),
	rhs(NULL), sol(NULL),
	fhat(NULL), ftmp(NULL), 
	rhsx(NULL), rhsy(NULL), rhsz(NULL),
	lambdax(NULL), lambday(NULL), lambdaz(NULL),
	initialized(false)
{
	const int *ncell = prob.cellnum;
	const double *hcell = prob.cellsize;

	// copy cell number and size
	for (int dir=0; dir<Ndim; dir++) {
		m_cellnum[dir] = ncell[dir];
		m_cellsize[dir] = hcell[dir];
	}

	// set BC
	for (int dir=0; dir<Ndim; dir++) {
		for (int side=0; side<=1; side++) {
			m_bctype[dir][side] = prob.bctype[dir][side];
		}
	}

	// set the best solver according to the BC type
	if (isPeriodic()) {
		m_solverType = DirSplitSolver;
	} else {
		m_solverType = NonSplitSolver;
	}

	// set thread number
#ifdef _OPENMP
	// if in OpenMP environment, use max possible threads
	m_threadNum = omp_get_max_threads();
#else
	m_threadNum = 1;
#endif

	// initialize FFT plan to NULL
	for (int dir=0; dir<Ndim; dir++) {
		fft_plan[dir][0] = NULL;
		fft_plan[dir][1] = NULL;
	}
}

//FastPoissonSolver3D(const int ncell[Ndim], const double hcell[Ndim])
//	: m_alpha(0.0), m_beta(1.0),
//	rhs(NULL), sol(NULL),
//	fhat(NULL), ftmp(NULL), 
//	rhsx(NULL), rhsy(NULL), rhsz(NULL),
//	lambdax(NULL), lambday(NULL), lambdaz(NULL),
//	initialized(false)
//{
//	for (int dir=0; dir<Ndim; dir++) {
//		m_cellnum[dir] = ncell[dir];
//		m_cellsize[dir] = hcell[dir];
//	}
//	
//	// by default set Neumann BC
//	for (int dir=0; dir<Ndim; dir++) {
//		m_bctype[dir][0] = BCType_NEU;
//		m_bctype[dir][1] = BCType_NEU;
//	}

//	for (int dir=0; dir<Ndim; dir++) {
//		fft_plan[dir][0] = NULL;
//		fft_plan[dir][1] = NULL;
//	}
//}


void FastPoissonSolver3D::initialize() {
	const int ndof = numDegreeOfFreedom();
	if (ndof <= 0) {
		std::cerr << __FUNCTION__ << ": problem invalid" << std::endl;
		exit(1);
	}

	const int nx = m_cellnum[0];
	const int ny = m_cellnum[1];
	const int nz = m_cellnum[2];

	// allocate buffers
	fhat = (double*) fftw_malloc(sizeof(double)*ndof);
	ftmp = (double*) fftw_malloc(sizeof(double)*ndof);

	rhsx = (double*) fftw_malloc(sizeof(double)*ndof);
	rhsy = (double*) fftw_malloc(sizeof(double)*ndof);
	rhsz = (double*) fftw_malloc(sizeof(double)*ndof);

	lambdax = (double*) fftw_malloc(sizeof(double)*nx);
	lambday = (double*) fftw_malloc(sizeof(double)*ny);
	lambdaz = (double*) fftw_malloc(sizeof(double)*nz);

	rhs = (double*) fftw_malloc(sizeof(double)*ndof);
	sol = (double*) fftw_malloc(sizeof(double)*ndof);

	//
	for (int dir=0; dir<Ndim; dir++) {
		fftw_r2r_kind fwd_kind; 
		double fwd_norm;
		selectTrans(true, m_cellnum[dir], m_cellsize[dir], m_bctype[dir][0], m_bctype[dir][1], fwd_kind, fwd_norm);
		fft_type[dir][0] = fwd_kind;
		fft_norm[dir][0] = fwd_norm;

		fftw_r2r_kind bwd_kind;
		double bwd_norm;
		selectTrans(false, m_cellnum[dir], m_cellsize[dir], m_bctype[dir][0], m_bctype[dir][1], bwd_kind, bwd_norm);
		fft_type[dir][1] = bwd_kind;
		fft_norm[dir][1] = bwd_norm;
	}

	// compute eigenvalue
	computeLambda(m_cellnum[0], m_cellsize[0], m_bctype[0][0], m_bctype[0][1], lambdax);
	computeLambda(m_cellnum[1], m_cellsize[1], m_bctype[1][0], m_bctype[1][1], lambday);
	computeLambda(m_cellnum[2], m_cellsize[2], m_bctype[2][0], m_bctype[2][1], lambdaz);

	// multi-thread FFT support
	if (fftw_init_threads() == 0) {
		std::cerr << "FFTW failed to initialize thread model" << std::endl;
		exit(1);
	}
	std::cout << "Number of threads = " << m_threadNum << std::endl;

	// find FFT plans
	if (m_solverType == DirSplitSolver) { // we have to solve dimension-by-dimension
		if (!isPeriodic()) { // if no P-BC exists, split solver is not the best
			std::cerr << __FUNCTION__
				<< ": Warning: No periodic BC, use NonSplitSolver is faster"
				<< std::endl;
		}

		init_dir_split();
	} else if (m_solverType == NonSplitSolver) { // use 3D FFT directly
		if (isPeriodic()) {
			std::cerr << __FUNCTION__
				<< ": Warning: For periodic BC, use DirSplitSolver is correct"
				<< std::endl;
		}

		init_non_split();
	} else {
		std::cerr << __FUNCTION__ << ": Invalid solver type" << std::endl;
		exit(1);
	}

	initialized = true;
} // FastPoissonSolver3D::initialize

void FastPoissonSolver3D::init_non_split() {
	const int nx = m_cellnum[0];
	const int ny = m_cellnum[1];
	const int nz = m_cellnum[2];
	const int nthreads = m_threadNum;

	fftw_plan_with_nthreads(nthreads);

	// RHS -> FHAT
	fftw_plan fwd_plan = fftw_plan_r2r_3d(
		nx, ny, nz, rhs, fhat, 
		(fftw_r2r_kind) fft_type[0][0], (fftw_r2r_kind) fft_type[1][0], (fftw_r2r_kind) fft_type[2][0],
		FFTW_MEASURE);

	// FHAT -> SOL
	fftw_plan bwd_plan = fftw_plan_r2r_3d(
		nx, ny, nz, fhat, sol,
		(fftw_r2r_kind) fft_type[0][1], (fftw_r2r_kind) fft_type[1][1], (fftw_r2r_kind) fft_type[2][1],
		FFTW_MEASURE);

	fft_plan[0][0] = fwd_plan;
	fft_plan[0][1] = bwd_plan;
}
void FastPoissonSolver3D::init_dir_split() {
	const int nx = m_cellnum[0];
	const int ny = m_cellnum[1];
	const int nz = m_cellnum[2];
	const int nthreads = m_threadNum;

	// forward transformations
	{ // RHS -> RHSX
		fftw_r2r_kind xfwd_kind = (fftw_r2r_kind) fft_type[0][0];

		fftw_plan xfwd_plan = NULL;
		if (xfwd_kind == FFTW_R2HC) {
			xfwd_plan = rfft3dPlanLastIndex(xfwd_kind, fhat, ftmp, ny, nz, nx, nthreads);
		} else {
			xfwd_plan = rfft3dPlanLastIndex(xfwd_kind, fhat, rhsx, ny, nz, nx, nthreads);
		}

		fft_plan[0][0] = xfwd_plan;
	}
	{ // RHSX -> RHSY
		fftw_r2r_kind yfwd_kind = (fftw_r2r_kind) fft_type[1][0];

		fftw_plan yfwd_plan = NULL;
		if (yfwd_kind == FFTW_R2HC) {
			yfwd_plan = rfft3dPlanLastIndex(yfwd_kind, fhat, ftmp, nz, nx, ny, nthreads);
		} else {
			yfwd_plan = rfft3dPlanLastIndex(yfwd_kind, fhat, rhsy, nz, nx, ny, nthreads);
		}

		fft_plan[1][0] = yfwd_plan;
	}
	{ // RHSY -> RHSZ
		fftw_r2r_kind zfwd_kind = (fftw_r2r_kind) fft_type[2][0];

		fftw_plan zfwd_plan = NULL;
		if (zfwd_kind == FFTW_R2HC) {
			zfwd_plan = rfft3dPlanLastIndex(zfwd_kind, fhat, ftmp, nx, ny, nz, nthreads);
		} else {
			zfwd_plan = rfft3dPlanLastIndex(zfwd_kind, fhat, rhsz, nx, ny, nz, nthreads);
		}

		fft_plan[2][0] = zfwd_plan;
	}

	// backward transformations
	{ // RHSZ -> RHSY
		fftw_r2r_kind zbwd_kind = (fftw_r2r_kind) fft_type[2][1];

		fftw_plan zbwd_plan = NULL;
		if (zbwd_kind == FFTW_HC2R) {
			zbwd_plan = rfft3dPlanLastIndex(zbwd_kind, ftmp, fhat, nx, ny, nz, nthreads);
		} else {
			zbwd_plan = rfft3dPlanLastIndex(zbwd_kind, rhsz, fhat, nx, ny, nz, nthreads);
		}

		fft_plan[2][1] = zbwd_plan;
	}

	{ // RHSY -> RHSX
		fftw_r2r_kind ybwd_kind = (fftw_r2r_kind) fft_type[1][1];

		fftw_plan ybwd_plan = NULL;
		if (ybwd_kind == FFTW_HC2R) {
			ybwd_plan = rfft3dPlanLastIndex(ybwd_kind, ftmp, fhat, nz, nx, ny, nthreads);
		} else {
			ybwd_plan = rfft3dPlanLastIndex(ybwd_kind, rhsy, fhat, nz, nx, ny, nthreads);
		}

		fft_plan[1][1] = ybwd_plan;
	}

	{ // RHSX -> SOL
		fftw_r2r_kind xbwd_kind = (fftw_r2r_kind) fft_type[0][1];

		fftw_plan xbwd_plan = NULL;
		if (xbwd_kind == FFTW_HC2R) {
			xbwd_plan = rfft3dPlanLastIndex(xbwd_kind, ftmp, fhat, ny, nz, nx, nthreads);
		} else {
			xbwd_plan = rfft3dPlanLastIndex(xbwd_kind, rhsx, fhat, ny, nz, nx, nthreads);
		}

		fft_plan[0][1] = xbwd_plan;
	}
} // FastPoissonSolver3D::init_dir_split


int FastPoissonSolver3D::solve(bool isHomogeneousBC) {
	if (!initialized) {
		std::cerr << __FUNCTION__ << ": solver not initialized" << std::endl;
		return 1;
	}

	if (isDegenerate()) {
		fix_rhs(rhs);
	}

	int ret = 1;
	if (m_solverType == DirSplitSolver) {
		ret = solve_dir_split();
	} else if (m_solverType == NonSplitSolver) {
		ret = solve_non_split();
	} else {
		std::cerr << __FUNCTION__ << ": solver type invalid" << std::endl;
	}

	return ret;
} // FastPoissonSolver3D::solve



int FastPoissonSolver3D::solve_non_split() {
	
	// RHS -> FHAT
	fftw_execute((fftw_plan) fft_plan[0][0]);

	// having FFT transformed data in FHAT
	// solve in eigenvalue space
	solve_diag(fhat);

	// FHAT -> SOL
	fftw_execute((fftw_plan) fft_plan[0][1]);

	// normalize
	normalize_result(sol);

	return 0;
} // solve_non_split

int FastPoissonSolver3D::solve_dir_split() {

	const int nx = m_cellnum[0];
	const int ny = m_cellnum[1];
	const int nz = m_cellnum[2];

	{ // x forward: RHS->RHSX
		// FWD rotate RHS(i,j,k) -> FHAT(j,k,i)
		rotateIndiceFwd(fhat, rhs, nx, ny, nz);

		// FFT FHAT(j,k,i) -> RHSX(j,k,i) 
		fftw_execute((fftw_plan) fft_plan[0][0]);

		if (fft_type[0][0] == FFTW_R2HC) {
			rfft3dPostR2HCLastIndex(ftmp, rhsx, ny, nz, nx);
		}

		normalize_result(rhsx, 0);
	}

	{ // y forward: RHSX->RHSY
		// FWD rotate RHSX(j,k,i) -> FHAT(k,i,j)
		rotateIndiceFwd(fhat, rhsx, ny, nz, nx);

		// FFT FHAT(k,i,j) -> RHSY(k,i,j)
		fftw_execute((fftw_plan) fft_plan[1][0]);

		if (fft_type[1][0] == FFTW_R2HC) {
			rfft3dPostR2HCLastIndex(ftmp, rhsy, nz, nx, ny);
		}

		normalize_result(rhsy, 1);
	}

	{ // z forward: RHSY->RHSZ
		// FWD rotate RHSY(k,i,j) -> FHAT(i,j,k)
		rotateIndiceFwd(fhat, rhsy, nz, nx, ny);

		// FFT FHAT(i,j,k) -> RHSZ(i,j,k)
		fftw_execute((fftw_plan) fft_plan[2][0]);

		if (fft_type[2][0] == FFTW_R2HC) {
			rfft3dPostR2HCLastIndex(ftmp, rhsz, nx, ny, nz);
		}

		normalize_result(rhsz, 2);
	}

	// having FFT transformed data in RHSZ
	// solve in eigenvalue space
	solve_diag(rhsz);

	{ // z backward, RHSZ->RHSY
		if (fft_type[2][1] == FFTW_HC2R) {
			rfft3dPreHC2RLastIndex(rhsz, ftmp, nx, ny, nz);
		}

		// FFT RHSZ(i,j,k) -> FHAT(i,j,k)
		fftw_execute((fftw_plan) fft_plan[2][1]);

		// BWD rotate FHAT(i,j,k) -> RHSY(k,i,j)
		rotateIndiceBwd(rhsy, fhat, nx, ny, nz);
	}

	{ // y backward, RHSY->RHSX
		if (fft_type[1][1] == FFTW_HC2R) {
			rfft3dPreHC2RLastIndex(rhsy, ftmp, nz, nx, ny);
		}

		// FFT RHSY(k,i,j) -> FHAT(k,i,j)
		fftw_execute((fftw_plan) fft_plan[1][1]);

		// BWD rotate FHAT(k,i,j) -> RHSX(j,k,i)
		rotateIndiceBwd(rhsx, fhat, nz, nx, ny);
	}

	{ // x backward, RHSX->SOL
		if (fft_type[0][1] == FFTW_HC2R) {
			rfft3dPreHC2RLastIndex(rhsx, ftmp, ny, nz, nx);
		}

		// FFT RHSX(j,k,i) -> FHAT(j,k,i)
		fftw_execute((fftw_plan) fft_plan[0][1]);

		// BWD rotate FHAT(j,k,i) -> SOL(i,j,k)
		rotateIndiceBwd(sol, fhat, ny, nz, nx);
	}

	return 0;
} // solve_dir_split

void FastPoissonSolver3D::fix_rhs(double *b) {
	double perturb = 0;

	const int ndof = numDegreeOfFreedom();
#pragma omp parallel for reduction(+: perturb)
	for (int idx=0; idx<ndof; idx++) {
		perturb += b[idx];
	}
	// std::cout << __FUNCTION__ << ": sum_rhs=" << perturb << std::endl;

	perturb /= (double) ndof;

	for (int idx=0; idx<ndof; idx++) {
		b[idx] -= perturb;
	}

	std::cout << __FUNCTION__  << ": perturb=" << perturb << std::endl;
} // FastPoissonSolver3D::fix_rhs

void FastPoissonSolver3D::solve_diag(double *diag) {
	const int nx = m_cellnum[0];
	const int ny = m_cellnum[1];
	const int nz = m_cellnum[2];

	const double alpha = m_alpha;

#	pragma omp parallel for collapse(2)
	for (int i=0; i<nx; i++) {
		for (int j=0; j<ny; j++) {
			const int offset = i*ny*nz + j*nz;
			for (int k=0; k<nz; k++) {
				int idx = offset + k;

				double lambdas = lambdax[i] + lambday[j] + lambdaz[k];
				diag[idx] /= (lambdas + alpha);
			}
		}
	}

	if (isDegenerate()) {
		diag[0] = 0;
	}
} // FastPoissonSolver3D::solve_diag

void FastPoissonSolver3D::normalize_result(double *res, int dir) {
	double norm = 1;
	if (dir == -1) {
		for (int dir=0; dir<Ndim; dir++) {
			for (int side=0; side<=1; side++) {
				norm *= fft_norm[dir][side];
			}
		}
	} else {
		norm = fft_norm[dir][0] * fft_norm[dir][1];
	}

	const int ndof = numDegreeOfFreedom();

#	pragma omp parallel for
	for (int idx=0; idx<ndof; idx++) {
		res[idx] *= norm;
	}
} // FastPoissonSolver3D::normalize_result


