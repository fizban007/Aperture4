FROM archlinux

USER root

RUN pacman -Syu --noconfirm --needed base base-devel cuda

RUN pacman -Syu --noconfirm --needed git cmake hdf5-openmpi boost ccls clang openssh

# install python
RUN pacman -Syu --noconfirm --needed python python-pip

# Install python-related packages
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install numpy scipy h5py toml matplotlib jupyter

# install latex
RUN pacman -Syu --noconfirm texlive-core texlive-science texlive-publishers texlive-latexextra texlive-formatsextra
RUN rm -r /var/cache/pacman

# finally define a normal user and switch to it
RUN useradd -r -g users -m developer
RUN mkdir /code && chown -R developer:users /code
USER developer
ENV PATH="${PATH}:/opt/cuda/bin