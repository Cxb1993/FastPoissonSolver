#pragma once



class TDMAParabolicSolver3D {
public:
	enum { Ndim = 3, };

	TDMAParabolicSolver3D(const int ncell[Ndim], const double hcell[Ndim]) 
		: m_alpha(1.0), m_beta(1.0)
	{
		for (int dir=0; dir<Ndim; dir++) {
			m_ncell[dir] = ncell[dir];
			m_hcell[dir] = hcell[dir];
		}
	}

	void setBeta(double beta) { m_beta = beta; }

	void initialize();

	double *getRhsData() { return m_rhs; }

	int solve();

	double *getSolData() { return m_sol; };

protected:

	void buildTriDiag(
		const int nx, const int ny, const int nz,
		const double dh,
		double *b, double *a, double *c) 
	{
		const double alpha = m_alpha;
		const double beta = m_beta;

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

	int tridiagSolve(const int n, 
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
	} // tridiagSolve

	int m_ncell[Ndim];
	double m_hcell[Ndim];

	double m_alpha;
	double m_beta;

	double *m_rhs;
	double *m_sol;

	double *m_phi[Ndim];

	double *m_adiag[Ndim];
	double *m_alo[Ndim];
	double *m_aup[Ndim];
	double *m_gamma;


};







