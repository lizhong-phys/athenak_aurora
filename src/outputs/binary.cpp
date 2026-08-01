//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file binary.cpp
//! \brief writes output data in binary format, which simply consists of each MeshBlock
//! written contiguously in order of "gid" in binary format.

#include <sys/stat.h>  // mkdir

#include <cstdio>      // fwrite(), fclose(), fopen(), fnprintf(), snprintf()
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm> // min

#include "athena.hpp"
#include "globals.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
// Constructor: also calls BaseTypeOutput base class constructor

MeshBinaryOutput::MeshBinaryOutput(ParameterInput *pin, Mesh *pm, OutputParameters op) :
  BaseTypeOutput(pin, pm, op) {
  // create directories for outputs
  // useful for mpiio-based outputs because on some supercomputers you may need to
  // set different stripe counts depending on whether mpiio is used in order to
  // achieve the best performance and not to crash the filesystem
  mkdir("bin",0775);
  bool single_file_per_rank = op.single_file_per_rank;
  if (single_file_per_rank) {
    char rank_dir[20];
    std::snprintf(rank_dir, sizeof(rank_dir), "bin/rank_%08d/", global_variable::my_rank);
    mkdir(rank_dir, 0775);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBinaryOutput:::WriteOutputFile(Mesh *pm)
//  \brief Cycles over all MeshBlocks and writes OutputData in binary format
//   All MeshBlocks are written to the same file.

void MeshBinaryOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  // check if slicing
  bool bin_slice = (out_params.slice1 || out_params.slice2 || out_params.slice3);

#if MPI_PARALLEL_ENABLED
  double stage_start = 0.0;
  double header_time = 0.0;
  double pack_time = 0.0;
  double data_time = 0.0;      // write time: root POSIX fwrite (slices) or MPI-IO write
  double close_time = 0.0;
  double split_time = 0.0;     // MPI_Comm_split for the per-slice communicator
  double gather_time = 0.0;    // MPI_Gatherv of slice blocks to the slice root
#endif

  // create filename: "bin/file_basename" + "." + "file_id" + "." + XXXXX + ".bin"
  // where XXXXX = 5-digit file_number

  bool single_file_per_rank = out_params.single_file_per_rank;
  std::string fname;
  if (single_file_per_rank) {
    // Generate a directory and filename for each rank
    char rank_dir[20];
    char number[7];
    std::snprintf(number, sizeof(number), ".%05d", out_params.file_number);
    std::snprintf(rank_dir, sizeof(rank_dir), "rank_%08d/", global_variable::my_rank);
    fname = std::string("bin/") + std::string(rank_dir) + out_params.file_basename
          + "." + out_params.file_id + number + ".bin";
  } else {
    // Existing behavior: single restart file
    char number[7];
    std::snprintf(number, sizeof(number), ".%05d", out_params.file_number);
    fname = std::string("bin/") + out_params.file_basename
          + "." + out_params.file_id + number + ".bin";
  }

  // ---- SLICE sub-communicator (permanent I/O improvement) -----------------------
  // For SLICE outputs, only ranks that own cells on the plane participate in the file's
  // collective open/write/close, via a per-slice communicator.  This keeps a single
  // flaky or non-owner node out of every slice's collective MPI_File_open (previously
  // all ranks opened every slice, though most never write it).  Full dumps: every rank
  // is active -> keep MPI_COMM_WORLD.  Per-rank files: independent I/O, no communicator.
  bool io_active = true;              // non-slice, or this rank owns slice cells
#if MPI_PARALLEL_ENABLED
  MPI_Comm slice_comm = MPI_COMM_WORLD;
  int slice_rank = global_variable::my_rank;
  int io_ranks = global_variable::nranks;   // # ranks that touch THIS file
  bool use_slice_comm = (bin_slice && !single_file_per_rank);
  bool use_gather = use_slice_comm;         // shared slices -> gather-to-root + POSIX
  if (use_slice_comm) {
    io_active = (outmbs.size() > 0);
    stage_start = MPI_Wtime();
    MPI_Comm_split(MPI_COMM_WORLD, io_active ? 1 : MPI_UNDEFINED,
                   global_variable::my_rank, &slice_comm);   // collective over WORLD
    split_time = MPI_Wtime() - stage_start;
    if (io_active) {
      MPI_Comm_rank(slice_comm, &slice_rank);
      MPI_Comm_size(slice_comm, &io_ranks);   // measured owner count (per slice)
    } else {
      io_ranks = 0;
    }
  }
  // the header is written once, by the first rank of the participating communicator
  bool write_header = single_file_per_rank || (io_active && slice_rank == 0);
#else
  bool write_header = true;
#endif

  IOWrapper binfile;   // used only by the MPI-IO / per-rank path below

  // Build the file header as ONE string.  Full dumps write it via MPI-IO (first rank);
  // gathered slices write it via POSIX on the slice root -- identical bytes either way.
#if MPI_PARALLEL_ENABLED
  stage_start = MPI_Wtime();
#endif
  std::string header_str;
  {
    std::stringstream msg;
    msg << "Athena binary output version=1.1" << std::endl
        // preheader size includes "size of preheader" line up to "number of variables"
        << "  size of preheader=5" << std::endl
        << "  time=" << pm->time << std::endl
        << "  cycle=" << pm->ncycle << std::endl
        << "  size of location=" << sizeof(Real) << std::endl
        << "  size of variable=" << sizeof(float) << std::endl
        << "  number of variables=" << outvars.size() << std::endl
        << "  variables:  ";
    for (int n=0; n<outvars.size(); n++) { msg << outvars[n].label.c_str() << "  "; }
    msg << std::endl;
    std::stringstream ost;
    pin->ParameterDump(ost);
    std::string sbuf=ost.str();
    msg << "  header offset=" << sbuf.size()*sizeof(char) << std::endl;
    header_str = msg.str() + sbuf;   // bytes identical to the old 3 sequential writes
  }
  std::size_t header_offset = header_str.size();
#if MPI_PARALLEL_ENABLED
  header_time = MPI_Wtime() - stage_start;
  stage_start = MPI_Wtime();
#endif

  //  5. Data.  An arbitrary number of scalars and vectors can be written (every element
  //  of the outvars vector), all in binary floats format

  int nout_vars = outvars.size();
  int nout_mbs = outmbs.size();
  int cells = 0;
  if (nout_mbs > 0) {
    int nout1 = outmbs[0].oie - outmbs[0].ois + 1;
    int nout2 = outmbs[0].oje - outmbs[0].ojs + 1;
    int nout3 = outmbs[0].oke - outmbs[0].oks + 1;
    cells = nout1*nout2*nout3;
  }

  // ois, oie, ojs, oje, oks, oke + il1, il2, il3, level +
  // x1min, x1max, x2min, x2max, x3min, x3max + data
  std::size_t data_size = 10*sizeof(int32_t) + 6*sizeof(Real)
                        + (cells*nout_vars)*sizeof(float);

  int ns_mbs = pm->gids_eachrank[global_variable::my_rank];
  int nb_mbs = pm->nmb_eachrank[global_variable::my_rank];

  // allocate 1D vector of floats used to convert and output data
  char *data = new char[nb_mbs*data_size];
  float *single_data = new float[cells];

  // Loop over MeshBlocks
  for (int m=0; m<nout_mbs; ++m) {
    char *pdata=&(data[m*data_size]);
    LogicalLocation loc = pm->lloc_eachmb[outmbs[m].mb_gid];
    int &ois = outmbs[m].ois;
    int &oie = outmbs[m].oie;
    int &ojs = outmbs[m].ojs;
    int &oje = outmbs[m].oje;
    int &oks = outmbs[m].oks;
    int &oke = outmbs[m].oke;

    // output indexing for MB
    int32_t nx = (int32_t)(ois);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);
    nx = (int32_t)(oie);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);
    nx = (int32_t)(ojs);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);
    nx = (int32_t)(oje);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);
    nx = (int32_t)(oks);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);
    nx = (int32_t)(oke);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);

    // logical location lx1, lx2, lx3
    nx = (int32_t)(loc.lx1);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);
    nx = (int32_t)(loc.lx2);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);
    nx = (int32_t)(loc.lx3);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);

    // physical refinement level
    nx = (int32_t)(loc.level-pm->root_level);
    memcpy(pdata,&(nx),sizeof(nx));
    pdata+=sizeof(nx);

    // coordinate location
    Real xv = outmbs[m].x1min;
    memcpy(pdata,&(xv),sizeof(xv));
    pdata+=sizeof(xv);
    xv = outmbs[m].x1max;
    memcpy(pdata,&(xv),sizeof(xv));
    pdata+=sizeof(xv);
    xv = outmbs[m].x2min;
    memcpy(pdata,&(xv),sizeof(xv));
    pdata+=sizeof(xv);
    xv = outmbs[m].x2max;
    memcpy(pdata,&(xv),sizeof(xv));
    pdata+=sizeof(xv);
    xv = outmbs[m].x3min;
    memcpy(pdata,&(xv),sizeof(xv));
    pdata+=sizeof(xv);
    xv = outmbs[m].x3max;
    memcpy(pdata,&(xv),sizeof(xv));
    pdata+=sizeof(xv);

    // output variables
    float tmp_data;
    for (int n=0; n<nout_vars; n++) {
      int cnt=0;
      for (int k=oks; k<=oke; k++) {
        for (int j=ojs; j<=oje; j++) {
          for (int i=ois; i<=oie; i++) {
            tmp_data = static_cast<float>(outarray(n,m,k-oks,j-ojs,i-ois));
            single_data[cnt] = tmp_data;
            cnt++;
          }
        }
      }
      memcpy(pdata,single_data,cells*sizeof(float));
      pdata+=cells*sizeof(float);
    }
  }

#if MPI_PARALLEL_ENABLED
  pack_time = MPI_Wtime() - stage_start;
  stage_start = MPI_Wtime();
#endif

  // ---------------------------------------------------------------------------
  // WRITE.  SHARED SLICES: gather each owner's packed blocks to the slice root, which
  // writes the whole file with POSIX -- no collective MPI_File_open on the shared slice
  // file (the recurring open-stall source).  Only remaining collective is MPI_Gatherv
  // over the OWNER ranks (comm, not filesystem).  FULL DUMPS + per-rank files keep the
  // MPI-IO / independent-POSIX path (bandwidth-bound; collective buffering wins there).
  // ---------------------------------------------------------------------------
#if MPI_PARALLEL_ENABLED
  if (use_gather) {
    if (io_active) {
      int local_bytes = static_cast<int>(data_size)*nout_mbs;
      std::vector<int> recvcounts, displs;
      char *gathered = nullptr;
      std::size_t total = 0;
      stage_start = MPI_Wtime();
      if (slice_rank == 0) { recvcounts.assign(io_ranks, 0); }
      MPI_Gather(&local_bytes, 1, MPI_INT,
                 (slice_rank == 0 ? recvcounts.data() : nullptr), 1, MPI_INT,
                 0, slice_comm);
      if (slice_rank == 0) {
        displs.assign(io_ranks, 0);
        for (int r = 0; r < io_ranks; ++r) {
          displs[r] = static_cast<int>(total);
          total += static_cast<std::size_t>(recvcounts[r]);
        }
        gathered = new char[total];
      }
      MPI_Gatherv(data, local_bytes, MPI_BYTE,
                  (slice_rank == 0 ? gathered : nullptr),
                  (slice_rank == 0 ? recvcounts.data() : nullptr),
                  (slice_rank == 0 ? displs.data() : nullptr),
                  MPI_BYTE, 0, slice_comm);
      gather_time = MPI_Wtime() - stage_start;
      if (slice_rank == 0) {
        stage_start = MPI_Wtime();
        FILE *fp = std::fopen(fname.c_str(), "wb");
        if (fp == nullptr) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "File '" << fname << "' could not be opened"
                    << std::endl;
          std::exit(EXIT_FAILURE);
        }
        std::fwrite(header_str.data(), 1, header_str.size(), fp);
        if (total > 0) { std::fwrite(gathered, 1, total, fp); }
        std::fclose(fp);
        data_time = MPI_Wtime() - stage_start;   // root POSIX write (filesystem)
        delete [] gathered;
      }
    }
  } else {
#endif
    // ---- MPI-IO / independent-POSIX single-file path -----------------------------
    binfile.Open(fname.c_str(), IOWrapper::FileMode::write, single_file_per_rank);
    if (write_header) {
      binfile.Write_any_type(header_str.c_str(), header_str.size(), "byte",
                             single_file_per_rank);
    }
#if MPI_PARALLEL_ENABLED
    stage_start = MPI_Wtime();
#endif
    if (bin_slice) {
      // per-rank slice file (single_file_per_rank): this rank writes only its blocks
      binfile.Write_any_type_at(data,(data_size*nout_mbs),header_offset,"byte",
                                single_file_per_rank);
    } else if (data_size*nb_mbs<=2147483648) {
      // full dump: collective write in parallel
      std::size_t myoffset = header_offset;
      if (!single_file_per_rank) { myoffset += data_size*ns_mbs; }
      binfile.Write_any_type_at_all(data,(data_size*nb_mbs),myoffset,"byte",
                                    single_file_per_rank);
    } else {
      // full dump > 2^31 bytes: write over each MeshBlock sequentially and in parallel
      noutmbs_max = pm->nmb_eachrank[0];
      noutmbs_min = pm->nmb_eachrank[0];
      for (int i=0; i<(global_variable::nranks); ++i) {
        noutmbs_max = std::max(noutmbs_max,pm->nmb_eachrank[i]);
        noutmbs_min = std::min(noutmbs_min,pm->nmb_eachrank[i]);
      }
      for (int m=0;  m<noutmbs_max; ++m) {
        char *pdata=&(data[m*data_size]);
        std::size_t myoffset = header_offset + data_size*m;
        if (!single_file_per_rank) { myoffset += data_size*ns_mbs; }
        if (m < noutmbs_min) {
          if (binfile.Write_any_type_at_all(pdata,(data_size),myoffset,"byte",
                                              single_file_per_rank) != data_size) {
            std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "binary data not written correctly to binary file, "
                << "binary file is broken." << std::endl;
            exit(EXIT_FAILURE);
          }
        } else if (m < pm->nmb_thisrank) {
          if (binfile.Write_any_type_at(pdata,(data_size),myoffset,"byte",
                                          single_file_per_rank) != data_size) {
            std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                 << std::endl << "binary data not written correctly to binary file, "
                 << "binary file is broken." << std::endl;
            exit(EXIT_FAILURE);
          }
        }
      }
    }
#if MPI_PARALLEL_ENABLED
    data_time = MPI_Wtime() - stage_start;
    stage_start = MPI_Wtime();
#endif
    binfile.Close(single_file_per_rank);
#if MPI_PARALLEL_ENABLED
    close_time = MPI_Wtime() - stage_start;
  }
#endif

#if MPI_PARALLEL_ENABLED
  // free the per-slice communicator (collective over the active ranks only)
  if (use_slice_comm && io_active) { MPI_Comm_free(&slice_comm); }

  // Gather per-rank stage timings and hostnames so rank 0 can report the slowest
  // participant in each stage without adding synchronization inside the stages.
  constexpr int nstages = 9;
  const double local_times[nstages] = {
      load_time, split_time, binfile.GetOpenTime(), binfile.GetTruncateTime(),
      header_time, pack_time, gather_time, data_time, close_time};
  char hostname[MPI_MAX_PROCESSOR_NAME] = {};
  int hostname_len = 0;
  MPI_Get_processor_name(hostname, &hostname_len);
  if (hostname_len >= MPI_MAX_PROCESSOR_NAME) {
    hostname[MPI_MAX_PROCESSOR_NAME-1] = '\0';
  } else {
    hostname[hostname_len] = '\0';
  }

  std::vector<double> all_times;
  std::vector<char> all_hostnames;
  if (global_variable::my_rank == 0) {
    all_times.resize(global_variable::nranks*nstages);
    all_hostnames.resize(global_variable::nranks*MPI_MAX_PROCESSOR_NAME);
  }
  MPI_Gather(local_times, nstages, MPI_DOUBLE,
             (global_variable::my_rank == 0 ? all_times.data() : nullptr),
             nstages, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
             (global_variable::my_rank == 0 ? all_hostnames.data() : nullptr),
             MPI_MAX_PROCESSOR_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);

  // measured participant count for THIS file (per-slice owner count; nranks for full)
  std::vector<int> all_io_ranks;
  if (global_variable::my_rank == 0) { all_io_ranks.resize(global_variable::nranks); }
  MPI_Gather(&io_ranks, 1, MPI_INT,
             (global_variable::my_rank == 0 ? all_io_ranks.data() : nullptr), 1,
             MPI_INT, 0, MPI_COMM_WORLD);

  if (global_variable::my_rank == 0) {
    int io_ranks_max = 0;
    for (int r = 0; r < global_variable::nranks; ++r) {
      io_ranks_max = std::max(io_ranks_max, all_io_ranks[r]);
    }
    const char *stage_names[nstages] = {
        "load", "split", "open", "truncate", "header", "pack", "gather", "data", "close"};
    std::ostringstream timing;
    timing << "BINARY_TIMING id=" << out_params.file_id
           << " file=" << out_params.file_number
           << " time=" << std::scientific << std::setprecision(6) << pm->time
           << " cycle=" << pm->ncycle
           << " io_ranks=" << io_ranks_max;
    for (int s=0; s<nstages; ++s) {
      int max_rank = 0;
      double max_time = all_times[s];
      for (int r=1; r<global_variable::nranks; ++r) {
        double rank_time = all_times[r*nstages+s];
        if (rank_time > max_time) {
          max_time = rank_time;
          max_rank = r;
        }
      }
      const char *max_hostname =
          &(all_hostnames[max_rank*MPI_MAX_PROCESSOR_NAME]);
      timing << " " << stage_names[s] << "_max=" << max_time
             << " " << stage_names[s] << "_rank=" << max_rank
             << " " << stage_names[s] << "_host=" << max_hostname;
    }
    std::cout << timing.str() << std::endl;
  }
#endif

  delete [] data;
  delete [] single_data;

  // increment counters
  out_params.file_number++;
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);

  return;
}
