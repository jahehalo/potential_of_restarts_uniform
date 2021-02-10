/**
* This is a modified version of probSAT SC13_v2.
* This version has the possibility to utilize Luby's restart strategy.
* Authors: Jan-Hendrik Lorenz and Julian Nickerl
* The original version of probSAT was developed by Adrian Balint.
*/


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/times.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <getopt.h>
#include <signal.h>
#include <string.h>

#define MAXCLAUSELENGTH 10000 //maximum number of literals per clause //TODO: eliminate this limit
#define STOREBLOCK  20000
# undef LLONG_MAX
#define LLONG_MAX  9223372036854775807
#define BIGINT long long int

void (*initLookUpTable)() = NULL;
void (*pickAndFlipVar)() = NULL;

/*--------*/

/*----Instance data (independent from assignment)----*/
/** The numbers of variables. */
int numVars;
/** The number of clauses. */
int numClauses;
/** The number of literals. */
int numLiterals;
/** The value of the variables. The numbering starts at 1 and the possible values are 0 or 1. */
char *atom;
char *startAssignment;
/** The clauses of the formula represented as: clause[clause_number][literal_number].
 * The clause and literal numbering start both at 1. literal and clause 0 0 is sentinel*/
int **clause;
/**min and max clause length*/
int maxClauseSize;
int minClauseSize;
/** The number of occurrence of each literal.*/
int *numOccurrence;
/** The clauses where each literal occurs. For literal i : occurrence[i+MAXATOMS][j] gives the clause =
 * the j'th occurrence of literal i.  */
int **occurrence;
int maxNumOccurences = 0; //maximum number of occurences for a literal
/*--------*/

/**----Assignment dependent data----*/
/** The number of false clauses.*/
int numFalse;
/** Array containing all clauses that are false. Managed as a list.*/
int *falseClause;
/** whereFalse[i]=j tells that clause i is listed in falseClause at position j.  */
int *whereFalse;
/** The number of true literals in each clause. */
unsigned short *numTrueLit;
/*the number of clauses the variable i will make unsat if flipped*/
int *breaks;
/** critVar[i]=j tells that for clause i the variable j is critically responsible for satisfying i.*/
int *critVar;
int bestVar;

/*----probSAT variables----*/
/** Look-up table for the functions. The values are computed in the initProbSAT method.*/
double *probsBreak;
/** contains the probabilities of the variables from an unsatisfied clause*/
double *probs;
double cb; //for break
double eps = 1.0;
int fct = 0; //function to use 0= poly 1=exp
int caching = 0;
/*--------*/

/*----Input file variables----*/
FILE *fp;
char *fileName;
/*---------*/

/** Run time variables variables*/BIGINT seed;
BIGINT maxTries = LLONG_MAX;
BIGINT maxFlips = LLONG_MAX;
BIGINT maxTotalFlips = LLONG_MAX;
BIGINT lubyStart = -1;
BIGINT flip;
BIGINT totalFlips;
BIGINT *restartTimes;
int totalNumberOfStarts = 0;
int *triesPerRestartTime;
int sizeRestartTimes = 0;
float timeOut = FLT_MAX;
int run = 1;
int printSol = 0;
double tryTime;
long ticks_per_second;
int bestNumFalse;
//parameters flags - indicates if the parameters were set on the command line
int cm_spec = 0, cb_spec = 0, fct_spec = 0, caching_spec = 0, assignment_spec=0;
BIGINT try = 0;

inline int abs(int a) {
	return (a < 0) ? -a : a;
}

void printFormulaProperties() {
	printf("\nc %-20s:  %s\n", "instance name", fileName);
	printf("c %-20s:  %d\n", "number of variables", numVars);
	printf("c %-20s:  %d\n", "number of literals", numLiterals);
	printf("c %-20s:  %d\n", "number of clauses", numClauses);
	printf("c %-20s:  %d\n", "max. clause length", maxClauseSize);
}

void printProbs() {
	int i;
	printf("c Probs values:\n");
	printf("c  ");
	for (i = 0; i <= 10; i++)
		printf(" %7i |", i);

	printf("\nc b");
	for (i = 0; i <= 10; i++) {
		if (probsBreak[i] != 0)
			printf(" %-6.5f |", probsBreak[i]);
	}
	printf("\n");
}

void printSolverParameters() {
	printf("\nc probSAT parameteres: \n");
	printf("c %-20s: %-20s\n", "using:", "only break");
	if (fct == 0)
		printf("c %-20s: %-20s\n", "using:", "polynomial function");
	else
		printf("c %-20s: %-20s\n", "using:", "exponential function");

	printf("c %-20s: %6.6f\n", "cb", cb);
	if (fct == 0) { //poly
		printf("c %-20s: %-20s\n", "function:", "probsBreak[break]*probsMake[make] = pow((eps + break), -cb);");
		printf("c %-20s: %6.6f\n", "eps", eps);
	} else { //exp
		printf("c %-20s: %-20s\n", "function:", "probsBreak[break]*probsMake[make] = pow(cb, -break);");
	}
	if (caching)
		printf("c %-20s: %-20s\n", "using:", "caching of break values");
	else
		printf("c %-20s: %-20s\n", "using:", "no caching of break values");
	//printProbs();
	printf("\nc general parameteres: \n");
	printf("c %-20s: %lli\n", "maxTries", maxTries);
	printf("c %-20s: %lli\n", "maxFlips", maxFlips);
	printf("c %-20s: %lli\n", "lubyStart", lubyStart);
	printf("c %-20s: %lli\n", "seed", seed);
	printf("c %-20s: \n", "-->Starting solver");
	fflush(stdout);
}

void printSolution() {
	register int i;
	printf("assignment: ");
	for (i = 1; i <= numVars; i++) {
		printf("%d", atom[i]);
	}
	printf("\n");

}

static inline void printStatsEndFlip() {
	if (numFalse < bestNumFalse) {
		//fprintf(stderr, "%8lli numFalse: %5d\n", flip, numFalse);
		bestNumFalse = numFalse;
	}
}

static inline void allocateMemory() {
	// Allocating memory for the instance data (independent from the assignment).
	numLiterals = numVars * 2;
	atom = (char*) malloc(sizeof(char) * (numVars + 1));
	clause = (int**) malloc(sizeof(int*) * (numClauses + 1));
	numOccurrence = (int*) malloc(sizeof(int) * (numLiterals + 1));
	occurrence = (int**) malloc(sizeof(int*) * (numLiterals + 1));
	critVar = (int*) malloc(sizeof(int) * (numClauses + 1));

	// Allocating memory for the assignment dependent data.
	falseClause = (int*) malloc(sizeof(int) * (numClauses + 1));
	whereFalse = (int*) malloc(sizeof(int) * (numClauses + 1));
	numTrueLit = (unsigned short*) malloc(sizeof(unsigned short) * (numClauses + 1));
}

static inline void parseFile() {
	register int i, j;
	int lit, r;
	int clauseSize;
	int tatom;
	char c;
	long filePos;
	fp = NULL;
	fp = fopen(fileName, "r");
	if (fp == NULL) {
		fprintf(stderr, "c Error: Not able to open the file: %s\n", fileName);
		exit(-1);
	}

	// Start scanning the header and set numVars and numClauses
	for (;;) {
		c = fgetc(fp);
		if (c == 'c') //comment line - skip content
			do {
				c = fgetc(fp); //read the complete comment line until a eol is detected.
			} while ((c != '\n') && (c != EOF));
		else if (c == 'p') { //p-line detected
			if ((fscanf(fp, "%*s %d %d", &numVars, &numClauses))) //%*s should match with "cnf"
				break;
		} else {
			printf("c No parameter line found! Computing number of atoms and number of clauses from file!\n");
			r = fseek(fp, -1L, SEEK_CUR); //try to unget c
			if (r == -1) {
				fprintf(stderr, "c Error: Not able to seek in file: %s", fileName);
				exit(-1);
			}
			filePos = ftell(fp);
			if (r == -1) {
				fprintf(stderr, "c Error: Not able to obtain position in file: %s", fileName);
				exit(-1);
			}

			numVars = 0;
			numClauses = 0;
			for (; fscanf(fp, "%i", &lit) == 1;) {
				if (lit == 0)
					numClauses++;
				else {
					tatom = abs(lit);
					if (tatom > numVars)
						numVars = tatom;
				}
			}
			printf("c numVars: %d numClauses: %d\n", numVars, numClauses);

			r = fseek(fp, filePos, SEEK_SET); //try to rewind the file to the beginning of the formula
			if (r == -1) {
				fprintf(stderr, "c Error: Not able to seek in file: %s", fileName);
				exit(-1);
			}

			break;
		}
	}
	// Finished scanning header.
	//allocating memory to use!
	allocateMemory();
	maxClauseSize = 0;
	minClauseSize = MAXCLAUSELENGTH;
	int *numOccurrenceT = (int*) malloc(sizeof(int) * (numLiterals + 1));

	int freeStore = 0;
	int *tempClause = 0;
	for (i = 0; i < numLiterals + 1; i++) {
		numOccurrence[i] = 0;
		numOccurrenceT[i] = 0;
	}

	for (i = 1; i <= numClauses; i++) {
		whereFalse[i] = -1;
		if (freeStore < MAXCLAUSELENGTH) {
			tempClause = (int*) malloc(sizeof(int) * STOREBLOCK);
			freeStore = STOREBLOCK;
		}
		clause[i] = tempClause;
		clauseSize = 0;
		do {
			r = fscanf(fp, "%i", &lit);
			if (lit != 0) {
				clauseSize++;
				*tempClause++ = lit;
				numOccurrenceT[numVars + lit]++;
			} else {
				*tempClause++ = 0; //0 sentinel as literal!
			}
			freeStore--;
		} while (lit != 0);
		if (clauseSize > maxClauseSize)
			maxClauseSize = clauseSize;
		if (clauseSize < minClauseSize)
			minClauseSize = clauseSize;
	}

	for (i = 0; i < numLiterals + 1; i++) {
		occurrence[i] = (int*) malloc(sizeof(int) * (numOccurrenceT[i] + 1));
		if (numOccurrenceT[i] > maxNumOccurences)
			maxNumOccurences = numOccurrenceT[i];
	}

	for (i = 1; i <= numClauses; i++) {
		j = 0;
		while ((lit = clause[i][j])) {
			occurrence[lit + numVars][numOccurrence[lit + numVars]++] = i;
			j++;
		}
		occurrence[lit + numVars][numOccurrence[lit + numVars]] = 0; //sentinel at the end!
	}
	probs = (double*) malloc(sizeof(double) * (numVars + 1));
	breaks = (int*) malloc(sizeof(int) * (numVars + 1));
	free(numOccurrenceT);
	fclose(fp);
}

static inline void init() {
	ticks_per_second = sysconf(_SC_CLK_TCK);
	register int i, j;
	int critLit = 0, lit;
	numFalse = 0;
	for (i = 1; i <= numClauses; i++) {
		numTrueLit[i] = 0;
		whereFalse[i] = -1;
	}

	for (i = 1; i <= numVars; i++) {
		atom[i] = rand() % 2;
		breaks[i] = 0;
	}
	//pass trough all clauses and apply the assignment previously generated
	for (i = 1; i <= numClauses; i++) {
		j = 0;
		while ((lit = clause[i][j])) {
			if (atom[abs(lit)] == (lit > 0)) {
				numTrueLit[i]++;
				critLit = lit;
			}
			j++;
		}
		if (numTrueLit[i] == 1) {
			//if the clause has only one literal that causes it to be sat,
			//then this var. will break the sat of the clause if flipped.
			critVar[i] = abs(critLit);
			breaks[abs(critLit)]++;
		} else if (numTrueLit[i] == 0) {
			//add this clause to the list of unsat caluses.
			falseClause[numFalse] = i;
			whereFalse[i] = numFalse;
			numFalse++;
		}
	}
}

static inline void initWithAssignment() {
	ticks_per_second = sysconf(_SC_CLK_TCK);
	register int i, j;
	int critLit = 0, lit;
	numFalse = 0;
	for (i = 1; i <= numClauses; i++) {
		numTrueLit[i] = 0;
		whereFalse[i] = -1;
	}

	for (i = 1; i <= numVars; i++) {
		atom[i] = startAssignment[i-1]-'0';
		breaks[i] = 0;
	}
	//pass trough all clauses and apply the assignment previously generated
	for (i = 1; i <= numClauses; i++) {
		j = 0;
		while ((lit = clause[i][j])) {
			if (atom[abs(lit)] == (lit > 0)) {
				numTrueLit[i]++;
				critLit = lit;
			}
			j++;
		}
		if (numTrueLit[i] == 1) {
			//if the clause has only one literal that causes it to be sat,
			//then this var. will break the sat of the clause if flipped.
			critVar[i] = abs(critLit);
			breaks[abs(critLit)]++;
		} else if (numTrueLit[i] == 0) {
			//add this clause to the list of unsat caluses.
			falseClause[numFalse] = i;
			whereFalse[i] = numFalse;
			numFalse++;
		}
	}
}



/** Checks whether the assignment from atom is a satisfying assignment.*/
static inline int checkAssignment() {
	register int i, j;
	int sat, lit;
	for (i = 1; i <= numClauses; i++) {
		sat = 0;
		j = 0;
		while ((lit = clause[i][j])) {
			if (atom[abs(lit)] == (lit > 0))
				sat = 1;
			j++;
		}
		if (sat == 0)
			return 0;
	}
	return 1;
}

//go trough the unsat clauses with the flip counter and DO NOT pick RANDOM unsat clause!!
// do not cache the break values but compute them on the fly (this is also the default implementation of WalkSAT in UBCSAT)
static inline void pickAndFlipNC() {
	register int i, j;
	int bestVar;
	int rClause, tClause;
	rClause = falseClause[flip % numFalse]; //random unsat clause
	bestVar = abs(clause[rClause][0]);
	double randPosition;
	int lit;
	double sumProb = 0;
	int xMakesSat = 0;
	i = 0;
	while ((lit = clause[rClause][i])) {
		breaks[i] = 0;
		//lit = clause[rClause][i];
		//numOccurenceX = numOccurrence[numVars - lit]; //only the negated occurrence of lit will count for break
		j=0;
		while ((tClause = occurrence[numVars - lit][j])){
			if (numTrueLit[tClause] == 1)
				breaks[i]++;
			j++;
		}
		probs[i] = probsBreak[breaks[i]];
		sumProb += probs[i];
		i++;
	}
	randPosition = (double) (rand()) / RAND_MAX * sumProb;
	for (i = i - 1; i != 0; i--) {
		sumProb -= probs[i];
		if (sumProb <= randPosition)
			break;
	}
	bestVar = abs(clause[rClause][i]);

	//flip bestvar
	if (atom[bestVar])
		xMakesSat = -bestVar; //if x=1 then all clauses containing -x will be made sat after fliping x
	else
		xMakesSat = bestVar; //if x=0 then all clauses containing x will be made sat after fliping x
	atom[bestVar] = 1 - atom[bestVar];
	//1. Clauses that contain xMakeSAT will get SAT if not already SAT
	//numOccurenceX = numOccurrence[numVars + xMakesSat];
	i = 0;
	while ((tClause = occurrence[xMakesSat + numVars][i])) {
		//if the clause is unsat it will become SAT so it has to be removed from the list of unsat-clauses.
		if (numTrueLit[tClause] == 0) {
			//remove from unsat-list
			falseClause[whereFalse[tClause]] = falseClause[--numFalse]; //overwrite this clause with the last clause in the list.
			whereFalse[falseClause[numFalse]] = whereFalse[tClause];
			whereFalse[tClause] = -1;
		}
		numTrueLit[tClause]++; //the number of true Lit is increased.
		i++;
	}
	//2. all clauses that contain the literal -xMakesSat=0 will not be longer satisfied by variable x.
	//all this clauses contained x as a satisfying literal
	//numOccurenceX = numOccurrence[numVars - xMakesSat];
	i = 0;
	while ((tClause = occurrence[numVars - xMakesSat][i])) {
		if (numTrueLit[tClause] == 1) { //then xMakesSat=1 was the satisfying literal.
			falseClause[numFalse] = tClause;
			whereFalse[tClause] = numFalse;
			numFalse++;
		}
		numTrueLit[tClause]--;
		i++;
	}
	//fliping done!
}
static inline void pickAndFlip() {
	int var;
	int rClause = falseClause[flip % numFalse];
	double sumProb = 0.0;
	double randPosition;
	register int i, j;
	int tClause; //temporary clause variable
	int xMakesSat; //tells which literal of x will make the clauses where it appears sat.
	i = 0;
	while ((var = abs(clause[rClause][i]))) {
		probs[i] = probsBreak[breaks[var]];
		sumProb += probs[i];
		i++;
	}
	randPosition = (double) (rand()) / RAND_MAX * sumProb;
	for (i = i - 1; i != 0; i--) {
		sumProb -= probs[i];
		if (sumProb <= randPosition)
			break;
	}
	bestVar = abs(clause[rClause][i]);

	if (atom[bestVar] == 1)
		xMakesSat = -bestVar; //if x=1 then all clauses containing -x will be made sat after fliping x
	else
		xMakesSat = bestVar; //if x=0 then all clauses containing x will be made sat after fliping x

	atom[bestVar] = 1 - atom[bestVar];

	//1. all clauses that contain the literal xMakesSat will become SAT, if they where not already sat.
	i = 0;
	while ((tClause = occurrence[xMakesSat + numVars][i])) {
		//if the clause is unsat it will become SAT so it has to be removed from the list of unsat-clauses.
		if (numTrueLit[tClause] == 0) {
			//remove from unsat-list
			falseClause[whereFalse[tClause]] = falseClause[--numFalse]; //overwrite this clause with the last clause in the list.
			whereFalse[falseClause[numFalse]] = whereFalse[tClause];
			whereFalse[tClause] = -1;
			critVar[tClause] = abs(xMakesSat); //this variable is now critically responsible for satisfying tClause
			//adapt the scores of the variables
			//the score of x has to be decreased by one because x is critical and will break this clause if fliped.
			breaks[bestVar]++;
		} else {
			//if the clause is satisfied by only one literal then the score has to be increased by one for this var.
			//because fliping this variable will no longer break the clause
			if (numTrueLit[tClause] == 1) {
				breaks[critVar[tClause]]--;
			}
		}
		//if the number of numTrueLit[tClause]>=2 then nothing will change in the scores
		numTrueLit[tClause]++; //the number of true Lit is increased.
		i++;
	}
	//2. all clauses that contain the literal -xMakesSat=0 will not be longer satisfied by variable x.
	//all this clauses contained x as a satisfying literal
	i = 0;
	while ((tClause = occurrence[numVars - xMakesSat][i])) {
		if (numTrueLit[tClause] == 1) { //then xMakesSat=1 was the satisfying literal.
			//this clause gets unsat.
			falseClause[numFalse] = tClause;
			whereFalse[tClause] = numFalse;
			numFalse++;
			//the score of x has to be increased by one because it is not breaking any more for this clause.
			breaks[bestVar]--;
			//the scores of all variables have to be increased by one ; inclusive x because flipping them will make the clause again sat
		} else if (numTrueLit[tClause] == 2) { //find which literal is true and make it critical and decrease its score
			j = 0;
			while ((var = abs(clause[tClause][j]))) {
				if (((clause[tClause][j] > 0) == atom[abs(var)])) { //x can not be the var anymore because it was flipped //&&(xMakesSat!=var)
					critVar[tClause] = var;
					breaks[var]++;
					break;
				}
				j++;
			}
		}
		numTrueLit[tClause]--;
		i++;
	}

}

double elapsed_seconds(void) {
	double answer;
	static struct tms prog_tms;
	static long prev_times = 0;
	(void) times(&prog_tms);
	answer = ((double) (((long) prog_tms.tms_utime) - prev_times)) / ((double) ticks_per_second);
	prev_times = (long) prog_tms.tms_utime;
	return answer;
}

static inline void printEndStatistics() {
	//printf("\nc EndStatistics:\n");
	if (numFalse != 0) {
		printf("timeout");
	}
	printf("%lli %lli\n", totalFlips, try);
}

static inline void printUsage() {
	printf("\n----------------------------------------------------------\n");
	printf("probSAT version SC13.2\n");
	printf("Authors: Adrian Balint\n");
	printf("Ulm University - Institute of Theoretical Computer Science \n");
	printf("2013\n");
	printf("----------------------------------------------------------\n");
	printf("\nUsage of probSAT:\n");
	printf("./probSAT [options] <DIMACS CNF instance> [<seed>]\n");
	printf("\nprobSAT options:\n");
	printf("which function to use:\n");
	printf("--fct <0,1> : 0 =  polynomial; 1 = exponential [default = 0]\n");
	printf("--eps <double_value> : eps>0 (only valid when --fct 0)[default = 1.0]\n");
	printf("which constant to use in the functions:\n");
	printf("--cb <double_value> : constant for break [default = k dependet]\n");
	printf("\nFurther options:\n");
	printf("--caching <0,1>, -c<0,1>  : use caching of break values \n");
	printf("--runs <int_value>, -r<int_value>  : maximum number of tries \n");
	printf("--maxflips <int_value> , -m<int_value>: number of flips per try \n");
	printf("--luby <int_value> , -L<int_value>: initial restart point if luby method is used \n");
	printf("--times <>, -i<>: string with (restartTimes, numTries) tuples separated by ;. Values of a tuple separated by ,");
	printf("--printSolution, -a : output assignment\n");
	printf("--help, -h : output this help\n");
	printf("----------------------------------------------------------\n\n");
}

void initPoly() {
	int i;
	probsBreak = (double*) malloc(sizeof(double) * (maxNumOccurences + 1));
	for (i = 0; i <= maxNumOccurences; i++) {
		probsBreak[i] = pow((eps + i), -cb);
	}
}

void initExp() {
	int i;
	probsBreak = (double*) malloc(sizeof(double) * (maxNumOccurences + 1));
	for (i = 0; i <= maxNumOccurences; i++) {
		probsBreak[i] = pow(cb, -i);
	}
}

void parseParameters(int argc, char *argv[]) {
	//define the argument parser
	static struct option long_options[] =
			{ { "fct", required_argument, 0, 'f' }, { "caching", required_argument, 0, 'c' },{"maxTotal", required_argument, 0, 'q'}, { "eps", required_argument, 0, 'e' }, { "cb", required_argument, 0, 'b' }, { "runs", required_argument, 0, 't' }, { "maxflips", required_argument, 0, 'm' }, { "luby", required_argument, 0, 'L' }, { "times", required_argument, 0, 'i' }, { "printSolution", no_argument, 0, 'a' }, { "help", no_argument, 0, 'h' },{ "assignment", required_argument, 0, 'l' },{ "flips", required_argument, 0, 'd' }, { 0, 0, 0, 0 } };

	while (optind < argc) {
		int index = -1;
		struct option * opt = 0;
		int result = getopt_long(argc, argv, "f:c:q:e:b:t:m:L:i:a:l:dh", long_options, &index); //
		if (result == -1)
			break; /* end of list */
		switch (result) {
		case 'h':
			printUsage();
			exit(0);
			break;
		case 'c':
			caching = atoi(optarg);
			caching_spec = 1;
			break;
		case 'f':
			fct = atoi(optarg);
			fct_spec = 1;
			break;
		case 'e':
			eps = atof(optarg);
			if (eps <= 0) {
				printf("\nERROR: eps should >0!!!\n");
				exit(0);
			}
			break;
		case 'q': // the deadline: After the total number of steps exceeds this number, the algorithm terminates
			maxTotalFlips = atoll(optarg);
			break;
		case 'b':
			cb = atof(optarg);
			cb_spec = 1;
			break;
		case 't': //maximum number of tries to solve the problems within the maxFlips
			maxTries = atoll(optarg);
			break;
		case 'm': //maximum number of flips to solve the problem
			maxFlips = atoll(optarg);
			break;
	    case 'L': //starting point if Luby method is used
	        lubyStart = atoi(optarg);
	        break;
		case 'a': //print assignment for variables at the end
			printSol = 1;
			break;
		case 'l':
			startAssignment = optarg;
			assignment_spec = 1;
			break;
		case 'd':
			flip = atoll(optarg);
			//printf("Start flips: %lli \n", flip);
			break;
	    case 'i':
	        restartTimes = (BIGINT *) malloc(sizeof(BIGINT)*100);
            triesPerRestartTime = (int *) malloc(sizeof(int)*100);
            
            char *tuple;
            int ind = 0;
            
            while((tuple = strsep(&optarg,";")) != NULL) {
                restartTimes[ind] = atoi(strsep(&tuple,","));
                triesPerRestartTime[ind] = atoi(strsep(&tuple,","));
                
                ind++;
            }

	        sizeRestartTimes = ind;
	        break;
	    
		case 0: /* all parameter that do not */
			/* appear in the optstring */
			opt = (struct option *) &(long_options[index]);
			printf("'%s' was specified.", opt->name);
			if (opt->has_arg == required_argument)
				printf("Arg: <%s>", optarg);
			printf("\n");
			break;
		default:
			printf("parameter not known!\n");
			printUsage();
			exit(0);
			break;
		}
	}
	if (optind == argc) {
		printf("ERROR: Parameter String not correct!\n");
		printUsage();
		exit(0);
	}
	fileName = *(argv + optind);

	if (argc > optind + 1) {
		seed = atoi(*(argv + optind + 1));
		if (seed == 0)
			printf("c there might be an error in the command line or is your seed 0?");
	} else
		seed = time(0);
}

void handle_interrupt() {
	printf("%s: %lli\n", "flips", flip);
	printSolution();
	fflush(NULL);
	exit(-1);
}

void setupSignalHandler() {
	signal(SIGTERM, handle_interrupt);
	signal(SIGINT, handle_interrupt);
	signal(SIGQUIT, handle_interrupt);
	signal(SIGABRT, handle_interrupt);
	signal(SIGKILL, handle_interrupt);
}

void setupParameters() {
	if (!caching_spec) {
		if (maxClauseSize <= 3){
			pickAndFlipVar = pickAndFlipNC; //no caching of the break values in case of 3SAT
			caching =0;
		}
		else{
			pickAndFlipVar = pickAndFlip; //cache the break values for other k-SAT
			caching = 1;
		}
	}
	else{
		if (caching)
			pickAndFlipVar = pickAndFlip; //cache the break values for other k-SAT
		else
			pickAndFlipVar = pickAndFlipNC; //no caching of the break values in case of 3SAT
	}
	if (!cb_spec) {
		if (maxClauseSize <= 3) {
			//cb = 2.06;
			//eps = 0.9;
			cb = 2.3;
			eps = 1.0;

		} else if (maxClauseSize <= 4)
			cb = 2.85;
		else if (maxClauseSize <= 5)
			cb = 3.7;
		else if (maxClauseSize <= 6)
			cb = 5.1;
		else
			cb = 5.4;
	}
	if (!fct_spec) {
		if (maxClauseSize < 4)
			fct = 0;
		else
			fct = 1;
	}
	if (fct == 0)
		initLookUpTable = initPoly;
	else
		initLookUpTable = initExp;
}

BIGINT nextRestart(int i) {
    BIGINT pot = 1;

    while(pot > 0){
        if (i == 2*pot -1) {
            return pot;
        } else if ( pot <= i && i < 2*pot -1) {
            return nextRestart(i-pot+1);
        } else {
            pot = pot * 2;   
        }
    }
    
    return 1;
}

int main(int argc, char *argv[]) {
	tryTime = 0.;
	double totalTime = 0.;
	flip = 0;
	parseParameters(argc, argv);
	parseFile();
	//printFormulaProperties();
	setupParameters(); //call only after parsing file!!!
	initLookUpTable(); //Initialize the look up table
	setupSignalHandler();
	//printSolverParameters();
	srand(seed);

	if(assignment_spec == 0)
		init();
	else
		initWithAssignment();
		
    totalFlips = 0;

    
	if (sizeRestartTimes != 0) {
	    // resList = 1;
	    maxTries = sizeRestartTimes;
	}
	int cont = 1;
	for (try = 0; (try < maxTries) && cont; ) {
		bestNumFalse = numClauses;
		
		// for(resListProgress = 0; resListProgress < resListCurrentMax; resListProgress++) {
		//     totalNumberOfStarts += 1;
		
		//     if (lub == 1) {
		//         maxFlips = nextRestart(try+1)*lubyStart;
		//     }
		//     if (resList == 1) {
		//         maxFlips = restartTimes[try];
		//         if (maxFlips < 0) { // this means we have reached a point where restarts are not helpful anymore
		//             maxFlips = LLONG_MAX;   
		//         }
		//         resListCurrentMax = triesPerRestartTime[try];
		        
		//         if (try == maxTries-1 && resListProgress == resListCurrentMax-1) {
		//             // We are in the last run, so from now on continue with luby-series!
		//             lub = 1;
		//             resList = 0;
		//             lubyStart = (BIGINT) maxFlips*(((double) maxFlips)/((double)restartTimes[try-1])) ;
		//             try = 0;
		//             maxTries = LLONG_MAX;
		//             resListCurrentMax = 1;
		            
		//         }
		//     }
		    
		    
		    //if (lub == 0) {
	        //    printf("Solving with %lli maxFlips\n", maxFlips);
	        //} else {
	    	//    printf("Solving with Luby and %lli maxFlips\n", maxFlips);
	        //}
		    if (totalFlips <= maxTotalFlips-maxFlips) {
			    for (; flip < maxFlips; flip++) {
				    if (numFalse == 0)
				    	break;
				    pickAndFlipVar();
				    printStatsEndFlip(); //update bestNumFalse
			    }
			    try++;
			} else {
				maxFlips = maxTotalFlips - totalFlips;
				for (; flip < maxFlips; flip++) {
				    if (numFalse == 0)
				    	break;
				    pickAndFlipVar();
				    printStatsEndFlip(); //update bestNumFalse
			    }
				try++;
				cont=0;
			}
		    tryTime = elapsed_seconds();
		    totalTime += tryTime;
		    if (numFalse == 0) {
			    if (!checkAssignment()) {
			        totalFlips += flip;
			    	fprintf(stderr, "c ERROR the assignment is not valid!");
			    	printf("c UNKNOWN");
			    	return 0;
			    } else {
			        totalFlips += flip;
			    	printEndStatistics();
			    	if (printSol == 1)
			    		printSolution();
			    	return 10;
			    }
		    } //else
			    //printf("c UNKNOWN best(%4d) current(%4d) (%-15.5fsec)\n", bestNumFalse, numFalse, tryTime);
		    totalFlips += flip;
		    flip = 0;
		    init();
		
		// }
	}
	printEndStatistics();
	return 0;
}

