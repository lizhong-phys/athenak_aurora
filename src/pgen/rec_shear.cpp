//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file rec_shear.cpp
//! \brief Problem generator for a double-layer (Harris) current sheet in an unstratified
//! MHD shearing box.
//!
//! STAGE 1 (this file).  This is deliberately a minimal edit of pgen/tests/mri3d.cpp --
//! a file already verified to run on Frontier.  Line for line it is mri3d with only the
//! field and density formulae replaced.  Everything mri3d does not do has been removed:
//!   * no user history function       (uses AthenaK's default MHD history)
//!   * no user source terms           (no vertical gravity yet)
//!   * no user boundary function
//!   * no parameter struct, no helper functions, no MPI calls, no Kokkos::fence,
//!     no parallel_reduce, no LeftEdgeX, no vector potential
//! The only Kokkos constructs here are the two par_for kernels and the random pool,
//! exactly as in mri3d.  If this still faults on Frontier, the problem generator is not
//! the cause and the initial DATA (or AthenaK's shearing box) is.
//!
//! Because there is no user history function, REMOVE these from your input file:
//!     <problem>/user_hist = true
//!     <output1>/user_hist_only = true
//! and drop <problem>/grav, sinkwidth, tau_sink, zero_net_momentum, beta_mri, nwx --
//! none of them are read by this version.
//!
//! Two current sheets sit at z = z_lower and z = z_upper.  Writing
//!     S(z) = tanh((z-z_upper)/jwidth) - tanh((z-z_lower)/jwidth) + 1
//! the field is
//!     Bx = bx2by*By0*S(z),   By = By0*S(z),   Bz = bx2by*bz2bx*By0
//! S(z) -> +1 at both z-boundaries and -> -1 midway between the sheets, so the setup is
//! periodic in z.  Density balances the magnetic pressure so total pressure is uniform:
//!     rho(z) = d0 + (By0/cs)^2 (1 + bx2by^2)/2 (1 - S(z)^2)
//! Random perturbations to the velocity are added in the initial conditions, as in HGB.
//!
//! REFERENCE: Hawley, J. F., Gammie, C.F. & Balbus, S. A., ApJ 440, 742-763 (1995).

// C headers
#include <algorithm>

// C++ headers
#include <cmath>      // sqrt()
#include <iostream>   // endl

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "shearing_box/shearing_box.hpp"
#include "pgen/pgen.hpp"

#include <Kokkos_Random.hpp>

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Sets a double Harris current sheet in an unstratified MHD shearing box

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;

  // First, do some error checks
  if (!(pmy_mesh_->three_d)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "rec_shear problem generator only works in 3D" << std::endl;
    exit(EXIT_FAILURE);
  }

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->pmhd != nullptr) {
    if (pmbp->pmhd->psbox_u == nullptr) {
      std::cout <<"### FATAL ERROR in "<< __FILE__ <<" at line " <<__LINE__ << std::endl
                << "Shearing box not enabled for rec_shear problem, likely missing "
                << "<shearing_box> block in input file" << std::endl;
      exit(EXIT_FAILURE);
    }
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "rec_shear problem generator only works in mhd" << std::endl;
    exit(EXIT_FAILURE);
  }

  EOS_Data &eos = pmbp->pmhd->peos->eos_data;
  if (eos.is_ideal) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "rec_shear problem generator only works with isothermal EOS"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // initialize problem variables
  Real amp    = pin->GetReal("problem","amp");
  Real beta   = pin->GetReal("problem","beta");
  Real d0     = pin->GetOrAddReal("problem","d0",1.0);
  Real jwidth = pin->GetOrAddReal("problem","jwidth",0.1);
  Real bx2by  = pin->GetOrAddReal("problem","bx2by",0.02);
  Real bz2bx  = pin->GetOrAddReal("problem","bz2bx",0.01);

  // background density, pressure, and magnetic field
  Real p0 = d0*SQR(eos.iso_cs);
  Real binit = std::sqrt(2.0*p0/beta);

  // locations of the two current sheets, at the quarter points of the z-domain
  Real zmin = pmy_mesh_->mesh_size.x3min;
  Real zmax = pmy_mesh_->mesh_size.x3max;
  Real zmid = 0.5*(zmin + zmax);
  Real zlwr = 0.5*(zmid + zmin);
  Real zupr = 0.5*(zmid + zmax);

  // density enhancement inside the sheets that keeps total pressure uniform
  Real dcoef = SQR(binit/eos.iso_cs)*(1.0 + SQR(bx2by))/2.0;

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;

  // Initialize magnetic field first, so entire arrays are initialized before adding
  // magnetic energy to conserved variables in next loop.  For 3D shearing box
  // B1=Bx, B2=By, B3=Bz.  Bx and By depend on z only and Bz is uniform, so div(B)=0.
  auto b0 = pmbp->pmhd->b0;
  par_for("rec_shear", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    Real sprof = tanh((x3v - zupr)/jwidth) - tanh((x3v - zlwr)/jwidth) + 1.0;
    Real bxv = binit*bx2by*sprof;
    Real byv = binit*sprof;
    Real bzv = binit*bx2by*bz2bx;

    b0.x1f(m,k,j,i) = bxv;
    b0.x2f(m,k,j,i) = byv;
    b0.x3f(m,k,j,i) = bzv;
    if (i==ie) b0.x1f(m,k,j,ie+1) = bxv;
    if (j==je) b0.x2f(m,k,je+1,i) = byv;
    if (k==ke) b0.x3f(m,ke+1,j,i) = bzv;
  });

  // Initialize conserved variables.  Density balances the magnetic pressure of the
  // sheets; random perturbations to the velocity seed reconnection.
  auto &u0 = pmbp->pmhd->u0;
  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);

  par_for("rec_shear-u", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    auto rand_gen = rand_pool64.get_state();  // get random number state this thread
    Real rval;

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    int nx3 = indcs.nx3;
    Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);

    // Set density, uniform in total (gas + magnetic) pressure
    Real sprof = tanh((x3v - zupr)/jwidth) - tanh((x3v - zlwr)/jwidth) + 1.0;
    Real rd = d0 + dcoef*(1.0 - SQR(sprof));
    u0(m,IDN,k,j,i) = rd;

    // Set momenta (with random perturbations to velocity).  NOTE no -q*Omega0*x term:
    // the background shear is carried by AthenaK's orbital advection.
    rval = amp*2.0*(rand_gen.frand() - 0.5);
    u0(m,IM1,k,j,i) = rd*rval;
    rval = amp*2.0*(rand_gen.frand() - 0.5);
    u0(m,IM2,k,j,i) = rd*rval;
    rval = amp*2.0*(rand_gen.frand() - 0.5);
    u0(m,IM3,k,j,i) = rd*rval;

    rand_pool64.free_state(rand_gen);  // free state for use by other threads
  });

  return;
}
