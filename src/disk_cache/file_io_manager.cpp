// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "file_io_manager.h"

#include <cassert>
#include <cstdio>
#include <iostream>
#include "tsl/robin_map.h"
#include "utils.h"
#define MAX_EVENTS 1024

namespace {
  typedef struct io_event io_event_t;
  typedef struct iocb     iocb_t;
}  // namespace

FileIOManager::~FileIOManager() {
  for (auto fd : this->fds) {
    int64_t ret;
    // check to make sure file_desc is closed
    ret = ::fcntl(fd, F_GETFD);
    if (ret == -1) {
      if (errno != EBADF) {
        std::cerr << "close() not called" << std::endl;
        // close file desc
        ret = ::close(fd);
        // error checks
        if (ret == -1) {
          std::cerr << "close() failed; returned " << ret << ", errno=" << errno
                    << ":" << ::strerror(errno) << std::endl;
        }
      }
    }
  }
}

io_context_t &FileIOManager::get_ctx() {
  std::unique_lock<std::mutex> lk(ctx_mut);
  // perform checks only in DEBUG mode
  if (ctx_map.find(std::this_thread::get_id()) == ctx_map.end()) {
    std::cerr << "bad thread access; returning -1 as io_context_t" << std::endl;
    return this->bad_ctx;
  } else {
    return ctx_map[std::this_thread::get_id()];
  }
}

void FileIOManager::register_thread() {
  auto                         my_id = std::this_thread::get_id();
  std::unique_lock<std::mutex> lk(ctx_mut);
  if (ctx_map.find(my_id) != ctx_map.end()) {
    std::cerr << "multiple calls to register_thread from the same thread"
              << std::endl;
    return;
  }
  io_context_t ctx = 0;
  int          ret = io_setup(MAX_EVENTS, &ctx);
  if (ret != 0) {
    lk.unlock();
    assert(errno != EAGAIN);
    assert(errno != ENOMEM);
    std::cerr << "io_setup() failed; returned " << ret << ", errno=" << errno
              << ":" << ::strerror(errno) << std::endl;
  } else {
    diskann::cout << "allocating ctx: " << ctx << " to thread-id:" << my_id
                  << std::endl;
    ctx_map[my_id] = ctx;
  }
  lk.unlock();
}

void FileIOManager::deregister_thread() {
  auto                         my_id = std::this_thread::get_id();
  std::unique_lock<std::mutex> lk(ctx_mut);
  assert(ctx_map.find(my_id) != ctx_map.end());

  lk.unlock();
  io_context_t ctx = this->get_ctx();
  io_destroy(ctx);
  //  assert(ret == 0);
  lk.lock();
  ctx_map.erase(my_id);
  std::cerr << "returned ctx from thread-id:" << my_id << std::endl;
  lk.unlock();
}

void FileIOManager::deregister_all_threads() {
  std::unique_lock<std::mutex> lk(ctx_mut);
  for (auto x = ctx_map.begin(); x != ctx_map.end(); x++) {
    io_context_t ctx = x.value();
    io_destroy(ctx);
    //  assert(ret == 0);
    //  lk.lock();
    //  ctx_map.erase(my_id);
    //  std::cerr << "returned ctx from thread-id:" << my_id << std::endl;
  }
  ctx_map.clear();
  //  lk.unlock();
}

int FileIOManager::open(const std::string &fname, const int flags) {
  int file_desc;
  if (O_CREAT & flags) {
    file_desc = ::open(fname.c_str(), flags, S_IRUSR | S_IWUSR);
  } else {
    file_desc = ::open(fname.c_str(), flags);
  }
  // error checks
  assert(file_desc != -1);
  if ( file_desc == -1) {
    std::cerr << "Opened file : " << fname << std::endl;
  }
  this->fds.push_back(file_desc);
  return this->fds.size() - 1;
}

void FileIOManager::close() {
  for (auto fd : this->fds) {
    // check to make sure file_desc is closed
    ::fcntl(fd, F_GETFD);

    ::close(fd);
  }
}

void FileIOManager::read(std::vector<AlignedRead> &read_reqs,
                                  io_context_t &ctx, int fid) {
  // assume all read from the same place
  assert(this->fds[fid] != -1);
  execute_io(ctx, this->fds[fid], read_reqs);
}

int FileIOManager::submit_read_reqs(std::vector<AlignedRead> &read_reqs,
                                        std::vector<int> &fids,
                                        io_context_t &ctx) {
  // assert(this->file_desc != -1);

  if (read_reqs.size() > MAX_EVENTS) {
    std::cerr << "The number of requests should not exceed " << MAX_EVENTS << std::endl;
    exit(-1);
  }
  int n_ops = read_reqs.size();
  std::vector<iocb_t *>    cbs(n_ops, nullptr);
  std::vector<struct iocb> cb(n_ops);
  for (int j = 0; j < n_ops; j++) {
    io_prep_pread(cb.data() + j, this->fds[fids[j]], read_reqs[j].buf,
                  read_reqs[j].len,
                  read_reqs[j].offset);
  }
  for (int i = 0; i < n_ops; i++) {
    // here annotate the read_req buffer, used in get_event.
    cb[i].data = (void*) read_reqs[i].buf;
    cbs[i] = cb.data() + i;
  }

  int ret = io_submit(ctx, (int64_t) n_ops, cbs.data());
  if (ret != n_ops) {
    std::cerr << "io_submit() failed; returned " << ret
              << ", expected=" << n_ops << ", ernno=" << errno << "="
              << ::strerror(-ret);
    std::cout << "ctx: " << ctx << "\n";
    exit(-1);
  }
  return n_ops;
}

int FileIOManager::submit_write_reqs(std::vector<AlignedWrite> &write_reqs,
                                        std::vector<int> &fids,
                                        io_context_t &ctx) {
  // assert(this->file_desc != -1);

  if (write_reqs.size() > MAX_EVENTS) {
    std::cerr << "The number of requests should not exceed " << MAX_EVENTS << std::endl;
    exit(-1);
  }
  int n_ops = write_reqs.size();
  std::vector<iocb_t *>    cbs(n_ops, nullptr);
  std::vector<struct iocb> cb(n_ops);
  for (int j = 0; j < n_ops; j++) {
    io_prep_pwrite(cb.data() + j, this->fds[fids[j]], write_reqs[j].buf,
                  write_reqs[j].len,
                  write_reqs[j].offset);
  }
  for (int i = 0; i < n_ops; i++) {
    // here annotate the read_req buffer, used in get_event.
    cb[i].data = (void*) write_reqs[i].buf;
    cbs[i] = cb.data() + i;
  }

  int ret = io_submit(ctx, (int64_t) n_ops, cbs.data());
  if (ret != n_ops) {
    std::cerr << "io_submit() failed; returned " << ret
              << ", expected=" << n_ops << ", ernno=" << errno << "="
              << ::strerror(-ret);
    std::cout << "ctx: " << ctx << "\n";
    exit(-1);
  }
  return n_ops;
}

void FileIOManager::get_events(IOContext& ctx, int n_ops) {
  std::vector<io_event_t> evts(n_ops);
  auto ret = io_getevents(ctx, (int64_t) n_ops, (int64_t) n_ops,
                          evts.data(), nullptr);
  if (ret != (int64_t) n_ops) {
    std::cerr << "io_getevents() failed; returned " << ret;
    exit(-1);
  }
}

int FileIOManager::get_events(IOContext& ctx, int min_r, int max_r, std::vector<char*>& ofts) {
  std::vector<io_event_t> evts(max_r);
  int64_t ret;
  if (min_r != 0) {
    ret = io_getevents(ctx, (int64_t) min_r, (int64_t) max_r, evts.data(), nullptr);
  }
  else {
    struct timespec ts = { 0, 0 } ;
    ret = io_getevents(ctx, (int64_t) min_r, (int64_t) max_r, evts.data(), &ts);
  }
  if (ret < min_r) {
    std::cerr << "io_getevents() failed; returned " << ret;
    exit(-1);
  }
  for (int i = 0; i < ret; i++) {
    if (evts[i].res2 != 0) {
      std::cerr << "io_getevents() failed; returned " << ret;
      exit(-1);
    }
    ofts[i] = (char*) evts[i].data;
  }
  return ret;
}
