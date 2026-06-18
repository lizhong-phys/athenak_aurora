//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file eos.cpp
//! \brief implements constructor and some fns for EquationOfState abstract base class

#include <float.h>
#include <string>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "eos/eos.hpp"

//----------------------------------------------------------------------------------------
// EquationOfState constructor

EquationOfState::EquationOfState(std::string bk, MeshBlockPack* pp, ParameterInput *pin) :
    pmy_pack(pp) {
  eos_data.dfloor = pin->GetOrAddReal(bk,"dfloor",(FLT_MIN));
  eos_data.pfloor = pin->GetOrAddReal(bk,"pfloor",(FLT_MIN));
  eos_data.tfloor = pin->GetOrAddReal(bk,"tfloor",(FLT_MIN));
  eos_data.sfloor = pin->GetOrAddReal(bk,"sfloor",(FLT_MIN));
  eos_data.sfloor1 = pin->GetOrAddReal(bk,"sfloor1",eos_data.sfloor);
  eos_data.sfloor2 = pin->GetOrAddReal(bk,"sfloor2",eos_data.sfloor);
  eos_data.rho1    = pin->GetOrAddReal(bk,"rho1",eos_data.dfloor);
  eos_data.rho2    = pin->GetOrAddReal(bk,"rho2",2*eos_data.dfloor);

  //
  eos_data.enable_r_dep_tfloor = pin->GetOrAddBoolean(bk,"enable_r_dep_tfloor",false);
  eos_data.r_tfloor            = pin->GetOrAddReal(bk,"r_tfloor", -1);
  eos_data.tfloor_local        = pin->GetOrAddReal(bk,"tfloor_local", -1);
  //
  eos_data.enable_sigma_tfloor = pin->GetOrAddBoolean(bk,"enable_sigma_tfloor",false);
  eos_data.sigma_tfloor1       = pin->GetOrAddReal(bk,"sigma_tfloor1", -1);
  eos_data.sigma_tfloor2       = pin->GetOrAddReal(bk,"sigma_tfloor2", -1);
  eos_data.sigma_tfloor_cap    = pin->GetOrAddReal(bk,"sigma_tfloor_cap", 1e10);
  eos_data.sigma1              = pin->GetOrAddReal(bk,"sigma1", -1);
  eos_data.sigma2              = pin->GetOrAddReal(bk,"sigma2", -1);
  
  eos_data.enable_sigma_dfloor = pin->GetOrAddBoolean(bk,"enable_sigma_dfloor",false);
  eos_data.sigma_max           = pin->GetOrAddReal(bk,"sigma_max",1.0e30);  // default: off

  // eos_data.sceiling            = pin->GetOrAddReal(bk,"sceiling", 1e30);
}

//----------------------------------------------------------------------------------------
//! \fn void ConsToPrim()
//! \brief No-Op versions of hydro and MHD conservative to primitive functions.
//! Required because each derived class overrides only one.

void EquationOfState::ConsToPrim(DvceArray5D<Real> &cons, DvceArray5D<Real> &prim,
                                 const bool only_testfloors,
                                 const int il, const int iu, const int jl, const int ju,
                                 const int kl, const int ku) {
}

void EquationOfState::ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                                 DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                                 const bool only_testfloors,
                                 const int il, const int iu, const int jl, const int ju,
                                 const int kl, const int ku) {
}

//----------------------------------------------------------------------------------------
//! \fn void PrimToCon()
//! \brief No-Op versions of hydro and MHD primitive to conservative functions.
//! Required because each derived class overrides only one.

void EquationOfState::PrimToCons(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons,
                                 const int il, const int iu, const int jl, const int ju,
                                 const int kl, const int ku) {
}
void EquationOfState::PrimToCons(const DvceArray5D<Real> &prim,
                                 const DvceArray5D<Real> &bcc, DvceArray5D<Real> &cons,
                                 const int il, const int iu, const int jl, const int ju,
                                 const int kl, const int ku) {
}
