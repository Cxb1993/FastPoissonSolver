#pragma once


/**
 * Boundary condition
 */
enum FastPoisson_BCType {
	BCType_PER,

	// staggered
	BCType_DIR,
	BCType_NEU,

	BCType_CDIR, // collocated
};

//enum FastPoisson_BCPair {
//	BCPair_PER_PER,
//	BCPair_DIR_DIR,
//	BCPair_DIR_NEU,
//	BCPair_NEU_DIR,
//	BCPair_NEU_NEU,
//};



class FastPoissonSolver3D {

public:
	enum { Ndim = 3, };

	enum SolverType {
		NonSplitSolver,
		DirSplitSolver,
	};

	struct PoissonProb {
		int cellnum[Ndim];
		double cellsize[Ndim];
		int bctype[Ndim][2];

		void setCellNumber(int dir, int ncell) {
			cellnum[dir] = ncell;
		}
		void setCellSize(int dir, double hcell) {
			cellsize[dir] = hcell;
		}
		void setBCType(int dir, int bc_lo, int bc_hi) {
			bctype[dir][0] = bc_lo;
			bctype[dir][1] = bc_hi;
		}
	};


	FastPoissonSolver3D(const PoissonProb &prob);

	virtual ~FastPoissonSolver3D() { /*TODO release memory*/ }

	void initialize();

	int numDegreeOfFreedom() const {
		int dof = 1;
		for (int dir=0; dir<Ndim; dir++) 
			dof *= m_cellnum[dir];
		return dof;
	}
	
	int isPeriodic(int dir) const {
		return m_bctype[dir][0]==BCType_PER && m_bctype[dir][1]==BCType_PER;
	}
	int isPeriodic() const {
		int is_per = 0;
		for (int dir=0; dir<Ndim; dir++) {
			if (isPeriodic(dir)) {
				is_per += 1;
			}
		}
		return is_per;
	}

	int isDegenerate() const {
		int is_degenerate = 0;
		if (m_alpha == 0) {
			is_degenerate = 1;
			for (int dir=0; dir<Ndim; dir++) {
				for (int side=0; side<=1; side++) {
					const int bc = m_bctype[dir][side];
					if (bc==BCType_DIR || bc==BCType_CDIR) {
						is_degenerate = 0;
					}
				}
			}
		}
		return is_degenerate;
	}
	
	void setBCType(int dir, int bc_lo, int bc_hi) {
		m_bctype[dir][0] = bc_lo;
		m_bctype[dir][1] = bc_hi;
	}

	void setAlpha(double alpha) { m_alpha = alpha; }
	void setBeta(double beta) { m_beta = beta; }
	
	double getAlpha() const { return m_alpha; }
	double getBeta() const { return m_beta; }

	void setSolverType(SolverType solver_type) {
		m_solverType = solver_type;
	}

	void setThreadNumber(int nthread) {
		m_threadNum = nthread;
	}


	double *getRhsData() { return rhs; }
	double *getSolData() { return sol; }

	int solve(bool isHomogeneousBC=true);

protected:

	void init_non_split();
	void init_dir_split();

	int solve_non_split();
	int solve_dir_split();

	void fix_rhs(double *b);
	void solve_diag(double *diag);
	void normalize_result(double *res, int dir=-1);

protected:
	int m_cellnum[Ndim];
	double m_cellsize[Ndim];

	int m_bctype[Ndim][2]; // DIM by (low,high)

	double m_alpha;
	double m_beta;

	SolverType m_solverType;
	int m_threadNum;

	bool initialized;

	// buffers
	double *fhat;
	double *ftmp;

	double *rhs;
	double *sol;

	double *rhsx;
	double *rhsy;
	double *rhsz;

	double *lambdax;
	double *lambday;
	double *lambdaz;

	// FFT related stuff
	
	// FFT type
	int fft_type[Ndim][2];
	// FFT normalizer
	double fft_norm[Ndim][2];

	// pointers to FFT classes
	void *fft_plan[Ndim][2];
};



