/*
 * Copyright (c) 2014, Vojtech Vasek
 *

 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.

 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.*
 *  
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
/*
 * ==============================================================================
 *
 *       Filename:  wave.c
 *
 *    Description:  This module is for work with WAV files.
 *                  It allows to read and write in this format.
 *                  This module defines structure ELEMENT, which is useful
 *                  for storing data from head of the WAV file, and enum
 *                  ENDIAN, which is useful for handeling data on byte-level.
 *
 *         Author:  Vojtech Vasek
 *
 * ==============================================================================
 */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <math.h>

#include "my_std.h"
#include "wave.h"
#include "complex.h"

/* Standard number of elements in WAV head */
#define HEADER_SIZE 13

/* Byte value of few strings findable in WAV head */
#define RIFF 0x52494646
#define RIFX 0x52494658
#define WAVE 0x57415645
#define FMT  0x666D7420
#define DATA 0x64617461

#define LESS_SET(a, b) if ((a) < (b)) { (b) = (a); }
#define MORE_SET(a, b) if ((a) > (b)) { (b) = (a); }

/* Standard WAV format header */
ELEMENT header[HEADER_SIZE] = {
	{BE, 0, 4, "ChunkID", 0},        /* 0; RIFF or RIFX, RIFX means that default is big endian */
	{LE, 4, 4, "ChunkSize", 0},      /* 1; */
	{BE, 8, 4, "Format", 0},         /* 2; should contain text "WAVE" = 57 41 56 45 */
	{BE, 12, 4, "Subchunk1ID", 0},   /* 3; should contain text "fmt" = 66 6D 74 20 */
	{LE, 16, 4, "Subchunk1Size", 0}, /* 4; 16 if PCM is used */
	{LE, 20, 2, "AudioFormat", 0},   /* 5; 1 ~ PCM, otherwise, there is some kind of compression in data chunk */
	{LE, 22, 2, "NumChannels", 0},   /* 6; 1 ~ mono, 2 ~ stereo */
	{LE, 24, 4, "SampleRate", 0},    /* 7; 44100 (Hz) is the most common */
	{LE, 28, 4, "ByteRate", 0},      /* 8; */
	{LE, 32, 2, "BlockAlign", 0},    /* 9; # of bytes for one sample */
	{LE, 34, 2, "BitsPerSample", 0}, /* 10; */
	{BE, 36, 4, "Subchunk2ID", 0},   /* 11; should contain text "data" = 64 61 74 61 */
	{LE, 40, 4, "Subchunk2Size", 0}  /* 12; # of bytes in the rest of the file */
};


/*
 *  Returns converted input int array in specified endian form
 *   into simple integer value.
 */
static unsigned long toInt(char *data, int size, ENDIAN endian) {
	unsigned long ret = 0;
	int i;
	for (i=0; i<size; i++) {
		/* Big Endian form */
		if (endian == 1) {
		  ret |= (unsigned char) data[i] & 0xFF;
			if (i != size-1) {
				ret <<= 8;
			}
		}
		/* Little Endian form */
	 	else {
			ret |= ((unsigned char)(data[i] & 0xFF) << (i*8));
		}
	}

	return ret;
}

/*
 *  Returns 1 if default endian is marked as Big, 0 otherwise.
 */
static ENDIAN getEndian(ELEMENT *h) {
	return (toInt(h[0].data, h[0].size, h[0].endian) == RIFX) ? BE : LE;
}

/*
 *  For given position and header, propriately calls toInt function.
 */
static unsigned long elementToInt(ELEMENT *h, int pos) {
	int endian = 0x01 & (getEndian(h) | h[pos].endian);
	return toInt(h[pos].data, h[pos].size, endian);
}

/*
 *  Conversion between array of short integers and double.
 *  Double is made by normalization into [-1; 1] interval.
 *
 *  short int *data  - input array with data to convert
 *  int size         - length of array *data
 */
static double toDouble(char *data, int size) {
	double ret = 0;
	if (size == 1) {
		ret = (double) ((unsigned char *) data)[0];
		ret = ret/255 - 0.5;
	} else if (size == 2) {
		ret = (double) ((char)data[0] + (char)data[1]*255);
		ret /= 65535;
	} else {
		fprintf(stderr, "Unsupported byte length\n");
	}

	return ret;
}

/*
 *  Returns number of channels as it is in given header.
 */
unsigned int getNumChannels(ELEMENT *h) {
	return elementToInt(h, 6);
}

/*
 *  Returns rate of samples in Hz as it is in given header.
 */
unsigned long getSampleRate(ELEMENT *h) {
	return elementToInt(h, 7);
}

/*
 *  Returns number of bytes in the data block.
 */
unsigned long getSubchunk2Size(ELEMENT *h) {
	return elementToInt(h, 12);
}

/*
 *  Compares data in element.data with given integer value
 *   and returns 0 if values are equal, 1 otherwise.
 */
static int elementComp(ELEMENT *h, int pos, unsigned int val) {
	if (elementToInt(h, pos) == val) {
		return 0;
	}
	return 1;
}

/*
 *  Retrieve data from WAV file and save them in local element
 *   structure "header".
 */
static int initHeader(int fd) {
	log_out(45, "WAV header data:\n");
	int i;
	for (i=0; i<HEADER_SIZE; i++) {
		int size = header[i].size;
 		if ((header[i].data = (char *) calloc((size+1), sizeof(char))) == NULL) {
			perror("calloc");
			return -1;
		}

		/* Move to specific offset where data should be stored */
		lseek(fd, header[i].offset, SEEK_SET);
		if (read(fd, header[i].data, size) != size) {
			fprintf(stderr, "Header was set incorrectly.");
			exit (2);
		}
		
		log_out(45, "%s = ", header[i].name);
		/* Use big endian */
		if (header[i].endian == BE) {
			log_out(45, "%s\n", header[i].data);
		} else {
			log_out(45, "%lu\n", elementToInt(header, i));
		}
	}
	log_out(45, "\n");

	/* Simple check, if file has WAVE header */
	if (elementComp(header, 2, WAVE) != 0) {
		fprintf(stderr, "Not a WAV file header\n");
		return -1;
	}
	/* Compression is not supported */
	if (elementToInt(header, 5) != 1) {
		fprintf(stderr, "Compression unsupported\n");
		return -1;
	}

	return 0;
}

/*
 *  Release memory allocated by ELEMENT structure and set
 *   its pointers to NULL.
 */
void freeHeader(ELEMENT *h) {
	int i;
	for (i=0; i<HEADER_SIZE; i++) {
		free(h[i].data);
		h[i].data = NULL;
	}
}

/*
 *  Returns "ch_id"-th sound channel from WAV file with file
 *   descriptor "fd".
 */
static C_ARRAY *getChannel(int fd, short ch_id) {
	/* Array for input samples */
	C_ARRAY *ca;
 	ca = allocCA(512);
	/* # of readed bytes */
	int r=0;
	/* # of bits per sample */
	int bps = elementToInt(header, 10);
	/* # of bytes per sample */
	int B_SIZE = bps/8;
	/* Jump to offset where the data starts */
	lseek(fd, 44 + B_SIZE*ch_id, SEEK_SET);
	/* # of channels */
	int nch = elementToInt(header, 6);
	/* # of total samples */
	int tns = elementToInt(header, 12);

	char *buf;
	if ((buf = (char *) calloc(B_SIZE, sizeof(char))) == NULL) {
		perror("calloc");
		return NULL;
	}
	/*
	 * At the end of file, there could be additional information,
	 * therefore we do not want to exceed # of samples for this channel
	 */
	while (read(fd, buf, B_SIZE) > 0 && r < tns/(B_SIZE*nch)) {
		/* Check if we need more memory for next values */
		if (ca->max == ca->len) {
			reallocCA(ca, get_pow(ca->len + 64, 2));
		}
		r++;

		ca->c[ca->len++].re = toDouble(buf, B_SIZE);

		if (r <= 11) {
			log_out(36, "%d-th sample: %.5f\n", r, ca->c[ca->len-1].re);
		}
		lseek(fd, B_SIZE*(nch-1), SEEK_CUR);
	}
	free(buf);

	return ca;
}

/*
 *  Takes pointers to file, where it tries to read first header
 *   then if successful, it allocates space in given C_ARRS for
 *   all channels in this WAV file and stores them separately as
 *   one sound track.
 */
ELEMENT *readWav(C_ARRS *cas, char *fpath) {
	int fd;

	if ((fd = open(fpath, O_RDONLY)) < 0) {
		perror("open");
		return NULL;
	}
	/* Read header data from file and store them */
	if (initHeader(fd) != 0) {
		fprintf(stderr, "Unacceptable WAVE header\n");
		close(fd);
		return NULL;
	}

	/* Store # of channels which is now in header variable */
	int nch = elementToInt(header, 6);

	/* Allocate space for all new input channels */
	if (cas->max - cas->len < nch) {
		reallocCAS(cas, cas->max + nch);
	}

	/* Now read all channels from the data section */
	int i;
	for (i=0; i<nch; i++) {
		log_out(36, "Channel %d:\n", i+1);
		cas->carrs[cas->len++] = getChannel(fd, i);
	}

	close(fd);
	return header;
}

/*
 *  Takes pointer to element structure header, and writes its values
 *   into given file.
 */
static void writeHeader(int fd, ELEMENT *h) {
	/* Header starts on the first bit of the file */
	lseek(fd, 0, SEEK_SET);

	int i;
	for (i=0; i<HEADER_SIZE; i++) {
		if (write(fd, h[i].data, h[i].size) <= 0) {
			perror("write");
			return;
		}
	}
}

/*
 *  Take double "d" from range [-1; 1] and convert it to "size" bytes,
 *   i.e. range [0, 2^(size*8) - 1].
 *  It's doing invers operation to the function toDouble.
 */
static void denormalize(char *buf, double d, int size, int endian) {
	/* Stores denormalized value of input double, 2 bytes are enough */
	short pom = 0;
	/* Denormalize to get integer value out of double */
	if (size == 1) {
		/* Shift by 1 byte */
		pom = (d + 0.5)*255; 
	} else if (size == 2) {
		/* Shift by 2 bytes */
		pom = d*65535;
	}

	int i;
	for (i=0; i<size; i++) {
		buf[i] = (char) ((pom >> i*8) & 0xff);
	}
}

/*
 *  Writes "ch_id"-th sound channel "ca" into data section of a file
 *   that has WAV header already prepared.
 */
int writeChannel(ELEMENT *h, C_ARRAY *ca, int fd, int ch_id) {
	int nch = elementToInt(h, 6);
	int bps = elementToInt(h, 10);
	int tot = elementToInt(h, 12);
	int B_SIZE = bps/8;         /* # of bytes per sample */
	int ctw = tot/(B_SIZE*nch); /* # of chunks (of size B_SIZE) to be written out */

	/*
	 *  Check if header was written out,
	 *   otherwise there would be missing chunk in this file.
	 */
	if (lseek(fd, 0, SEEK_CUR) < 44) {
		fprintf(stderr, "Header file was not written yet, can't write data\n");
		return -1;
	}
	char *buf; /* Buffer for the output bytes */
	if ((buf = (char *) calloc(B_SIZE, sizeof(char))) == NULL) {
		perror("calloc");
		return -1;
	}

	/* Jump to the start of the data section */
	lseek(fd, 44+B_SIZE*ch_id, SEEK_SET);
	/* Start writing sound data */
	int i;
	for (i=0; i<ctw && i<ca->len; i++) {
		denormalize(buf, ca->c[i].re, B_SIZE, 0);
		if (write(fd, buf, B_SIZE) <= 0) {
			perror("write");
			return -1;
		}
		lseek(fd, B_SIZE*(nch-1), SEEK_CUR);
	}
	free(buf);

	return 0;
}

/*
 *  Writes array *cas into file using WAV format.
 *  Struct element *h must be already prepared.
 */
void writeWav(ELEMENT *h, C_ARRS *cas, char *fpath) {
	int fd;
	int nch = getNumChannels(h);

	/* Check if we have the same # of channels as its in header */
	if (nch != cas->len) {
		fprintf(stderr, "Header file improperly set, %d channels required, got %d.\n", nch, cas->len);
		return;
	}
	/* Create if not exists, rewrite existing file */
	if ((fd = open(fpath, O_WRONLY | O_CREAT | O_TRUNC, 0755)) < 0) {
		perror("open");
		return;
	}

	/* Only LE is supported */
	strcpy(h[0].data, "RIFF");
	writeHeader(fd, h);

	/* Write all given channels */
	int i;
	for (i=0; i<nch; i++) {
		printf("Writing out %d. channel...\n", i+1);
		writeChannel(h, cas->carrs[i], fd, i);
	}

	close(fd);
}
