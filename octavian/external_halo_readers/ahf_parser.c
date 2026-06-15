// REVIEW: 
// It took a long time to write this, and I needed extensive help from Claude.
// I would not be able to debug it but hopefully someone who understands C can take a look.
// This should be regarded as entirely experimental, please do not use it for data analysis!

// libraries
#include <stdio.h>   // file operations: fopen, fgets, fclose
#include <stdlib.h>  // general utilities

// NOTE: this is a C implementation of the state machine

long parse_ahf_particles(
    const char *filename,   // string (path to file)
    long *out_hids,         // pointer to array of halo IDs (pre-allocated from Python)
    long *out_pids,         // pointer to array of particle IDs
    char *out_ptypes,       // pointer to array of ptype codes
    int *valid_ptypes,      // pointer to array of valid ptype codes {0, 1, 4, 5}
    int n_valid             // how many valid ptypes
) 
{
    FILE *f = fopen(filename, "r");  // "r" = read mode
    if (f == NULL) return -1;        // if file doesn't exist, return error
    char line[256];          // buffer to hold a line as we read in (256 characters, should be enough)
    long remaining = 0;      // for the state machine: particles in halo
    long current_hid = 0;    // current halo ID
    long idx = 0;            // index position in output array
    long a, b;               // a: nparticles, b: hid
    int i, valid;            // loop counter and flag

    while (fgets(line, sizeof(line), f)) {         // fgets reads one line from the file into the line buffer (like a for loop)
        if (sscanf(line, "%ld %ld", &a, &b) != 2) // sscanf scans the string line to extract values from it according to the specified pattern
            continue; // so if there are not two entries on the line, continue
        if (remaining == 0) { // if remaining is 0 we have hit a header line
        // header line: (nparticles | haloID)
        remaining = a; 
        current_hid = b;
        continue; // move on to next line
        }

        remaining--; // C equivalent of python remaining -= 1
        valid = 0;
        for (i = 0; i < n_valid; i++) { // check the four possible ptypes
            if ((int)b == valid_ptypes[i]) {
                valid = 1;
                break; // particle is one found in the ptype map
            }
        }
        if (!valid) continue; // if valid is 0, this particle is not Octavian-friendly and is skipped
        out_hids[idx] = current_hid;
        out_pids[idx] = a;
        out_ptypes[idx] = (char)b;
        idx++; // increment our write position (move on to next one when writing in)
    }
    fclose(f); // must explicitly close the file 
    return idx; // number of particles we wrote into the array
}