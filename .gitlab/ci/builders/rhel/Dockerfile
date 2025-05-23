#
# Copyright 2013-2025 Software Radio Systems Limited
#
# By using this file, you agree to the terms and conditions set
# forth in the LICENSE file which can be found at the top level of
# the distribution.
#

ARG RHEL_VERSION
ARG CR_REGISTRY_URI
FROM ${CR_REGISTRY_URI}/ci/rhel-images/srsgnb-ubi8:pre-builder_${RHEL_VERSION}

ENV PATH=/opt/rh/gcc-toolset-12/root/usr/bin:$PATH

ADD install_dependencies.sh /usr/local/bin
RUN chmod +x /usr/local/bin/install_dependencies.sh

# Workaround for gcc/g++ path in builder.sh
RUN install_dependencies.sh && \
    rm -f /usr/bin/gcc && rm -f /usr/bin/g++ && \
    ln -s /opt/rh/gcc-toolset-12/root/usr/bin/gcc /usr/bin/gcc && \
    ln -s /opt/rh/gcc-toolset-12/root/usr/bin/g++ /usr/bin/g++

# Install gtest
RUN cd /tmp && git clone https://github.com/google/googletest.git && \
    cd googletest && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j $(nproc) && make -j $(nproc) install && \
    cd ../../ && rm -rf googletest

# TODO: install python (including venv) to support changed_tests and ram_reporter scripts

COPY builder.sh /usr/local/bin/builder.sh
