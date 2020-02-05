//
// Created by David Seery on 13/08/2013.
// Copyright (c) 2016 University of Sussex. All rights reserved.
// @contributor: David Seery <D.Seery@sussex.ac.uk>
//
//

#ifndef CPPTRANSPORT_SPLINE_H
#define CPPTRANSPORT_SPLINE_H


#include <vector>
#include <array>
#include <limits>
#include <type_traits>
#include <cassert>
#include <memory>

#include "datatable.h"
#include "bspline.h"
#include "bsplinebuilder.h"


namespace transport
  {

    template <typename number>
    class spline1d
      {
        
        // TYPES
        
      public:
        
        //! spline y-values are 'number'
        using value_type = number;
        
        //! spline x-values are 'double'
        using domain_type = double;

        
        // CONSTRUCTOR, DESTRUCTOR

      public:

        //! set up a 1d spline object with given data points
        spline1d(const std::vector<domain_type>& x, const std::vector<value_type>& y);

        //! set up a default constructor
        spline1d() = default;

        //! destructor is default
        ~spline1d() = default;

        
        // INTERFACE -- OFFSETS

      public:

        void set_spline_data(const std::vector<domain_type>& x, const std::vector<value_type>& y, const int spline_order=5);

		    //! get current offset
		    value_type get_offset() const { return(this->offset); }

		    //! set new offset
		    void set_offset(value_type o) { this->offset = o; }


        // INTERFACE -- EVALUATE VALUES

      public:

        //! evaluate the spline at a given value of x
        value_type eval(domain_type x);

        //! evaluate the spline at a given value of x
        value_type operator()(domain_type x) { return this->eval(x); }


        // INTERFACE -- EVALUATE DERIVATIVES

      public:

        //! evaluate the derivative of the spline at a given value of x
        value_type eval_diff(domain_type x);


		    // INTERFACE -- MISC DATA

      public:

		    //! get minimum x value
		    domain_type get_min_x() const { return(this->min_x); }

		    //! get maximum x value
		    domain_type get_max_x() const { return(this->max_x); }


        // WRITE TO STREAM

      public:

        //! write self to stream
        void write(std::ostream& out) const;


        // INTERNAL DATA


      protected:

        //! pointer to spline object; its lifetime is managed by std::shared_ptr,
        //! which also deals correctly with taking copies.
        //! This is a raw BSpline, meaning that it goes through each of the data points.
        //! We use it to interpolate values.
        std::shared_ptr<SPLINTER::BSpline> b_spline;

        //! pointer to penalized-B spline object.
        //! This is smoothed, so it need not go through every data point.
        //! We use it to interpolate derivatives
        std::shared_ptr<SPLINTER::BSpline> p_spline;

        //! DataTable instance
        SPLINTER::DataTable table;

		    //! offset to apply on evaluation
		    value_type offset;

		    //! number of points
		    size_t N;

		    //! minimum x value
		    domain_type min_x;

		    //! maximum x value
		    domain_type max_x;

        //! is initiated?
        bool init = false; 

      };


    template <typename number>
    spline1d<number>::spline1d(const std::vector<domain_type>& x, const std::vector<value_type>& y)
      : offset(0.0),
        min_x(std::numeric_limits<domain_type>::max()),
        max_x(-std::numeric_limits<domain_type>::max())
      {
        this->set_spline_data(x, y);
      }

    template <typename number>
    void spline1d<number>::set_spline_data(const std::vector<domain_type>& x, const std::vector<value_type>& y, const int spline_order)
      {
        offset = 0.0;
        min_x = std::numeric_limits<domain_type>::max();
        max_x = -std::numeric_limits<domain_type>::max();
        
        assert(x.size() == y.size());

        N  = x.size();

        SPLINTER::DataTable new_table;

        // copy supplied data into Splinter DataTable
        for(unsigned int i = 0; i < N; ++i)
          {
            // y may need an explicit downcast to domain_type
            new_table.addSample(x[i], static_cast<domain_type>(y[i]));

		        if(x[i] < min_x) min_x = x[i];
		        if(x[i] > max_x) max_x = x[i];
          }

        try
	        {
            b_spline = std::make_shared<SPLINTER::BSpline>(SPLINTER::BSpline::Builder(new_table).degree(spline_order).build());
            p_spline = std::make_shared<SPLINTER::BSpline>(SPLINTER::BSpline::Builder(new_table).degree(spline_order).smoothing(SPLINTER::BSpline::Smoothing::PSPLINE).alpha(0.03).build());
	        }
				catch(SPLINTER::Exception& xe)
					{
						throw std::runtime_error(xe.what());
					}
      }


    template <typename number>
    typename spline1d<number>::value_type spline1d<number>::eval(domain_type x)
      {
        SPLINTER::DenseVector xv(1);
        xv(0) = x;

        return( static_cast<value_type>(this->b_spline->eval(xv)) - this->offset );
      }


    template <typename number>
    typename spline1d<number>::value_type spline1d<number>::eval_diff(domain_type x)
      {
        SPLINTER::DenseVector xv(1);
        xv(0) = x;

        return( static_cast<value_type>((this->p_spline->evalJacobian(xv))(0)) );
      }


    template <typename number>
    void spline1d<number>::write(std::ostream& out) const
      {
        for(auto t = this->table.cbegin(); t != this->table.cend(); ++t)
          {
            assert(t->getDimX() == 1);
            out << "x = " << t->getX()[0] << ", y = " << t->getY() << '\n';
          }
      }


    template <typename number, typename Char, typename Traits>
    std::basic_ostream<Char, Traits>& operator<<(std::basic_ostream<Char, Traits>& out, const spline1d<number>& obj)
      {
        obj.write(out);
        return(out);
      };



  }   // namespace transport


#endif // CPPTRANSPORT_SPLINE_H
